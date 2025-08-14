## QLoRA 微調 Llama-3-8B 進行同業競爭偵測 (Competitor Detection)

#### 專案概述 (Project Overview)

本專案旨在利用大型語言模型（LLM）的強大自然語言理解能力，自動分析多來源的文本資料（如客戶服務紀錄、會議摘要、業務日報等），以達成以下目標：
偵測：判斷文本中是否提及到有任何同業經營的證據。
提取：如果提及，則準確提取出同業的具體名稱，以及能支持此判斷的原文佐證句子。
結構化輸出：將分析結果以標準化的 JSON 格式輸出，方便後續的數據分析、商業智能（BI）報表製作或自動化流程。
本專案最終採用對 meta-llama/Meta-Llama-3-8B-Instruct 基礎模型進行 QLoRA 微調的技術路徑，以在有限的硬體資源下，實現高效且精準的模型訓練與推論。



#### 技術棧 (Technology Stack)

**基礎模型:** meta-llama/Meta-Llama-3-8B-Instruct
**微調方法:** QLoRA (4-bit NormalFloat Quantization)
**核心框架:**
- PyTorch
- transformers
- peft (Parameter-Efficient Fine-Tuning)
- accelerate
- bitsandbytes

**開發環境:** Python 3.9+, CUDA

#### 使用方法 (Usage)
- 在window的環境下直接下載 bitsandbytes會報錯，請找到適配自己cuda的下載源(當下是下載了bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl)
0. 準備訓練資料：包含正面範例、負面範例和邊緣案例，這樣模型才能學會區分其中的差別。

1. 模型訓練
腳本: train_llama3.py
功能: 讀取訓練資料，對 Llama-3 8B 模型進行 QLoRA 微調，並將訓練好的 LoRA 適配器權重保存下來。
執行 `train.py` 訓練完成後，LoRA 適配器會被保存在 ./llama3-8b-competitor-final 資料夾中。訓練過程中的檢查點會保存在 ./llama3-8b-competitor-finetune。

2. 接續斷點訓練
如果您需要中斷訓練後繼續，可以修改 train.py 腳本中的 trainer.train()：
trainer.train(resume_from_checkpoint=True)

3. 批次預測
腳本: predict_batch.py
功能: 讀取一個 CSV 或 Excel 檔案，對其中的 text 欄位進行逐行預測，並將結果（同業競爭, 同業名稱, 原文）附加到新的欄位後，保存為一個新的 CSV 檔案。

#### 專案結構 (Project Structure)

```
├── Meta-Llama-3-8B-Instruct/      # 本地存放的基礎模型
├── llama3-8b-competitor-final/    # 訓練後生成的 LoRA 適配器
├── llama3-8b-competitor-finetune/ # 訓練過程中的檢查點
│
├── train.py               # 訓練腳本
├── predict.py             # 批次預測腳本
│
├── trainData.json                   # 訓練資料範例
├── input_data.csv                   # 批次預測的輸入資料範例
├── data_with_predictions.csv        # 批次預測的輸出結果
│
└── README.md                  
```

#### 未來可拓展方向 (Future Work)
- 擴充資料集：持續收集更多樣化、edge case 來提升模型的泛化能力。
- 模型評估：建立一個獨立的測試集，並編寫評估腳本，以量化指標（如精確率、召回率、F1 分數）來評估模型的輸出品質。
- 超參數搜索：使用 Optuna 或 Ray Tune 等工具，系統性地尋找最佳的學習率、LoRA rank 等超參數。
