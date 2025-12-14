import torch
import pytest
from main import Net  # main.py から Net クラスをインポート

def test_model_structure():
    """モデルの出力サイズが正しいかテストする"""
    model = Net()
    
    # ダミーの入力データを作成 (バッチサイズ1, 1チャンネル, 28x28画像)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # モデルに入力
    output = model(dummy_input)
    
    # 検証: 出力の形が (1, 10) であること
    assert output.shape == (1, 10)

def test_device():
    """CUDAが使えるか、CPUかを確認する（エラーが出ないかの確認）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device in ["cuda", "cpu"]