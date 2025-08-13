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
待補


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
