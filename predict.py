import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)
import json
import pandas as pd
from tqdm import tqdm


#  1. 設定模型路徑
base_model_path = "./Meta-Llama-3-8B-Instruct"
# LoRA 適配器路徑 (Llama-3 訓練的輸出)
adapter_path = "./llama3-8b-competitor-final" 

#  2. 設定設備與量化
device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

#  3. 載入與訓練時完全相同的基礎模型和 Tokenizer 
print(f"正在載入基礎模型: {base_model_path}...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#  4. 融合為 Llama-3 訓練的 LoRA 適配器 
print("正在載入並融合 LoRA 適配器...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("模型準備完成！")

def predict_competitor(text_input):
    instruction = "分析以下文本，判斷顧客是否有被同業競爭者經營。若有，請提供同業名稱和佐證句子，並以JSON格式輸出。JSON應包含 '同業競爭' (布林值), '同業名稱' (列表), 和 '原文' (列表) 三個欄位。"
    
    # 使用 Llama-3 的官方提示模板
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant designed to analyze text and output structured JSON.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n\n文本內容：\n{text_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            do_sample=True,
            temperature=0.2, # 對於 JSON 生成，可以嘗試更低的溫度
            top_p=0.9,
            max_new_tokens=256,
            # Llama-3 的結束標記有兩個，都需要提供
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )

    # 解碼時不再跳過特殊標記，以便手動處理
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    try:
        assistant_part = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        final_output = assistant_part.split(tokenizer.eos_token)[0].strip()
        if final_output.startswith("```json"):
            final_output = final_output[7:]
        if final_output.endswith("```"):
            final_output = final_output[:-3]
        return json.loads(final_output)
    except Exception as e:
        
        # 將錯誤訊息和原始輸出作為返回字典的一部分
        return {
            "error": f"解析失敗: {e}", 
            "raw_output": response_text,
            "original_input": text_input
        }


def predict_competitor_with_certainty(text_input):
    if not isinstance(text_input, str) or not text_input.strip():
        return {"同業競爭": False, "同業名稱": [], "原文": [], "certainty": 1.0, "error": "Invalid input"}

    instruction = "分析以下文本，判斷顧客是否有被同業競爭者經營。若有，請提供同業名稱和佐證句子，並以JSON格式輸出。JSON應包含 '同業競爭' (布林值), '同業名稱' (列表), 和 '原文' (列表) 三個欄位。"
    
    # 構建完整的 Prompt
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant designed to analyze text and output structured JSON.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n\n文本內容：\n{text_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # --- 階段一：計算「同業競爭」的確定性分數 (獨立計算) ---
    # 這裡我們只截取到可能出現 "true" 或 "false" 的位置
    prompt_for_bool_score = full_prompt + '{"同業競爭": ' 
    
    inputs_bool = tokenizer(prompt_for_bool_score, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs_bool = model(**inputs_bool) # 使用模型前向傳播
        
    last_token_logits = outputs_bool.logits[0, -1, :] # 取最後一個 token 的 logits
    
    true_token_id = tokenizer.convert_tokens_to_ids("true")
    false_token_id = tokenizer.convert_tokens_to_ids("false")

    if true_token_id == tokenizer.unk_token_id or false_token_id == tokenizer.unk_token_id:
        return {"同業競爭": False, "同業名稱": [], "原文": [], "certainty": 0.5, "error": "Cannot find true/false tokens in vocab"}

    logits_for_softmax = torch.tensor([last_token_logits[false_token_id], last_token_logits[true_token_id]]).to(device)
    probabilities = F.softmax(logits_for_softmax, dim=0)
    
    prob_true = probabilities[1].item()
    predicted_bool_prelim = prob_true > 0.5 # 初步判斷
    certainty_score = prob_true if predicted_bool_prelim else 1 - prob_true # 計算確定性

    # --- 階段二：讓模型生成完整的 JSON 字串 ---
    inputs_full_gen = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_outputs = model.generate(
            **inputs_full_gen,
            do_sample=True,
            temperature=0.2, # 為了 JSON 結構穩定，溫度可以稍低
            top_p=0.9,
            max_new_tokens=256,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 解碼完整的生成結果
    response_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 部分
    try:
        assistant_start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_start_tag not in response_text:
            raise ValueError("Assistant tag not found in response.")
        
        # 提取 assistant 部分，並去掉後面的 EOS token
        assistant_part = response_text.split(assistant_start_tag)[1].strip()
        # Llama-3 的 EOS token 可能會在末尾
        if assistant_part.endswith(tokenizer.eos_token):
            assistant_part = assistant_part[:-len(tokenizer.eos_token)].strip()
        if assistant_part.endswith("<|eot_id|>"): # 另一個 EOS token
            assistant_part = assistant_part[:-len("<|eot_id>")].strip()

        # 進一步清理：有時模型會輸出 ```json...```
        if assistant_part.startswith("```json"):
            assistant_part = assistant_part[7:].strip()
        if assistant_part.endswith("```"):
            assistant_part = assistant_part[:-3].strip()

        # 嘗試解析 JSON
        parsed_json = json.loads(assistant_part)
        
        # 確保解析後的 JSON 包含我們期望的欄位，並將其格式化
        final_prediction = {
            "同業競爭": parsed_json.get("同業競爭", False),
            "同業名稱": list(map(str, parsed_json.get("同業名稱", []))),
            "原文": list(map(str, parsed_json.get("原文", []))),
            "certainty": certainty_score # 填充之前計算的確定性分數
        }
        return final_prediction

    except Exception as e:
        # 如果解析失敗，返回錯誤信息和確定性分數
        return {
            "同業競爭": False, # 預設為 False
            "同業名稱": [], 
            "原文": [], 
            "certainty": certainty_score, # 即使解析失敗，這個分數依然有效
            "error": f"解析最終JSON失敗: {e}", 
            "raw_output": response_text, 
            "original_input": text_input
        }



if __name__ == "__main__":
    input_file = '同業競爭_data.xlsx' 
    output_file = '同業_with_predictions.xlsx'
    uncertain_samples_file = '同業_retrain_samples.xlsx'
    
    print(f"正在從 {input_file} 讀取數據...")
    try:
        df = pd.read_excel(input_file)
        # 為了快速測試，可以只取前100行
        df = df[:10] 
    except FileNotFoundError:
        print(f"[錯誤] 輸入檔案 {input_file} 未找到！請檢查檔案路徑。")
        exit()

    # --- 準備新欄位 ---
    df.loc[:, '同業競爭'] = None
    df.loc[:, '同業名稱'] = None
    df.loc[:, '原文'] = None
    df.loc[:, 'certainty'] = None 
    df.loc[:, 'status'] = 'SUCCESS' # 預設狀態

    failed_cases = [] # 收集錯誤案例

    original_log_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()

    print("開始進行批次預測...")
    # 確保調用新的函數
    for row in tqdm(df.itertuples(), total=len(df), desc="預測進度"):
        memo_text = getattr(row, 'likememo', '')
        prediction_result = predict_competitor_with_certainty(memo_text) # <--- 調用新的函數
        
        if "error" in prediction_result:
            failed_cases.append({
                "index": row.Index,
                "text": memo_text,
                "error_message": prediction_result["error"],
                "raw_output": prediction_result.get("raw_output", "N/A"),
                "certainty_at_error": prediction_result.get("certainty", 0.0) # 記錄出錯時的 certainty
            })
            df.loc[row.Index, 'status'] = "ERROR"
            # 填入錯誤時的預設值和記錄的 certainty
            df.loc[row.Index, '同業競爭'] = prediction_result.get('同業競爭', False) 
            df.loc[row.Index, '同業名稱'] = None
            df.loc[row.Index, '原文'] = None
            df.loc[row.Index, 'certainty'] = prediction_result.get('certainty', 0.0)
            continue
            
        # 填入成功解析的結果
        df.loc[row.Index, '同業競爭'] = prediction_result.get('同業競爭')
        # 列表轉字符串時，確保元素都是字符串類型，否則map(str, ...)可能會失敗
        df.loc[row.Index, '同業名稱'] = ','.join(prediction_result.get('同業名稱', [])) or None
        df.loc[row.Index, '原文'] = ' || '.join(prediction_result.get('原文', [])) or None
        df.loc[row.Index, 'certainty'] = prediction_result.get('certainty')

    hf_logging.set_verbosity(original_log_level)

    print("\n\n批次預測全部完成！")    

    if failed_cases:
        print("-" * 50)
        print(f"[報告] 總共遇到 {len(failed_cases)} 筆預測或解析失敗的案例。")
        for failure in failed_cases[:min(5, len(failed_cases))]: # 顯示前5個或所有失敗案例
            print(f"  - DataFrame 索引 {failure['index']}:")
            print(f"    錯誤原因: {failure['error_info']}")
            print(f"    原始輸入: '{str(failure['text'])[:80]}...'")
            print(f"    原始模型輸出: '{str(failure['raw_output'])[:80]}...'") # 打印原始輸出方便除錯
        if len(failed_cases) > 5:
            print(f"  ... 以及其他 {len(failed_cases) - 5} 個錯誤未顯示。")
        print("您可以在輸出的 Excel 檔案中，透過篩選 'status' 欄位為 'ERROR' 來找到所有失敗的案例。")
        print("-" * 50)
    else:
        print("[報告] 所有樣本均成功預測且解析成功！")

    # --- 提取最不確定的樣本 (與之前邏輯相同) ---
    successful_df = df[df['status'] == 'SUCCESS'].copy()
    # 確保 certainty 是數值類型，避免排序報錯
    successful_df['certainty'] = pd.to_numeric(successful_df['certainty'], errors='coerce') 
    successful_df_sorted = successful_df.sort_values(by='certainty', ascending=True) # 不確定性是 certainty 越低越不確定
    
    num_samples_for_labeling = 50
    uncertain_samples = successful_df_sorted.head(num_samples_for_labeling)
    
    if not uncertain_samples.empty:
        print(f"\n提取了 {len(uncertain_samples)} 筆最不確定的樣本，準備進行人工標註...")
        print(f"這些樣本的 certainty 分數範圍: {uncertain_samples['certainty'].min():.4f} - {uncertain_samples['certainty'].max():.4f}")
        print(f"正在將這些樣本保存至 {uncertain_samples_file}...")
        uncertain_samples.to_excel(uncertain_samples_file, index=False)
    else:
        print("\n沒有成功計算的樣本可供提取。")

    # --- 保存完整結果 ---
    print(f"\n正在將帶有分數的完整結果保存至 {output_file}...")
    df.to_excel(output_file, index=False)
    
    print("所有處理已完成！")
