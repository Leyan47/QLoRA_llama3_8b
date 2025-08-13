import json
import pandas as pd
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)

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


if __name__ == "__main__":
    # --- 1. 讀取數據 ---
    input_file = 'predict_data.xlsx' 
    output_file = 'data_with_predictions.csv'
    
    print(f"正在從 {input_file} 讀取數據...")
    try:
        df = pd.read_excel(input_file)
        df = df[:100]
    except FileNotFoundError:
        print(f"[錯誤] 輸入檔案 {input_file} 未找到！請檢查檔案路徑。")
        exit()

    # --- 2. 準備新欄位 ---
    # 使用 .loc 來避免 SettingWithCopyWarning
    df.loc[:, '同業競爭'] = None
    df.loc[:, '同業名稱'] = None
    df.loc[:, '原文'] = None

    # 在循環開始前，將日誌級別設為 "ERROR" 
    original_log_level = hf_logging.get_verbosity() # 保存原始的日誌級別
    hf_logging.set_verbosity_error() # 只顯示錯誤，忽略所有警告和資訊

    print("開始進行批次預測...")
    failed_predictions = []
    
    # 這個循環現在將在一個非常「安靜」的環境下運行
    for row in tqdm(df.itertuples(), total=len(df), desc="預測進度"):
        memo_text = getattr(row, 'likememo', '')
        prediction = predict_competitor(memo_text)
        
        if "error" in prediction:
            failed_predictions.append({
                "index": row.Index,
                "error_info": prediction
            })
            df.loc[row.Index, '同業競爭'] = "PREDICTION_ERROR"
            continue
            
        df.loc[row.Index, '同業競爭'] = prediction.get('同業競爭', False)
        competitor_names = prediction.get('同業名稱', [])
        df.loc[row.Index, '同業名稱'] = ','.join(map(str, competitor_names)) if competitor_names else None
        evidence_sentences = prediction.get('原文', [])
        df.loc[row.Index, '原文'] = ' || '.join(map(str, evidence_sentences)) if evidence_sentences else None

    # 在循環結束後，恢復原始的日誌級別
    hf_logging.set_verbosity(original_log_level)

    print("\n\n批次預測全部完成！")    

    #  在循環結束後，統一報告問題 
    print("\n\n批次預測全部完成！")
    if failed_predictions:
        print("-" * 50)
        print(f"[報告] 總共遇到 {len(failed_predictions)} 筆解析失敗的案例。")
        for failure in failed_predictions[:5]: # 只顯示前 5 個案例的詳細資訊
            print(f"  - DataFrame 索引 {failure['index']}:")
            print(f"    錯誤原因: {failure['error_info']['error']}")
            print(f"    原始輸入: '{failure['error_info']['original_input'][:80]}...'")
        if len(failed_predictions) > 5:
            print(f"  ... 以及其他 {len(failed_predictions) - 5} 個錯誤未顯示。")
        print("您可以在輸出的 CSV 檔案中，透過篩選 'PREDICTION_ERROR' 來找到所有失敗的案例。")
        print("-" * 50)
    else:
        print("[報告] 所有樣本均成功預測且解析成功！")


    print(f"\n預測完成！正在將結果保存至 {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("所有處理已完成！")
