import os
from openai import OpenAI

# --- 設定 ---

# 1. OpenRouter APIキーを設定してください。
# 環境変数 'OPENROUTER_API_KEY' から読み込むことを推奨します。
# os.environ.get('OPENROUTER_API_KEY', 'YOUR_API_KEY_HERE') の形式で設定できます。
# 環境変数に設定していない場合は、'YOUR_API_KEY_HERE'の部分に直接キーを貼り付けてください。
openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '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')

# OpenRouterのAPIベースURL
# OpenRouterはOpenAI互換の形式を採用しているため、エンドポイントを指定します。
openrouter_base_url = "https://openrouter.ai/api/v1"

# 使用するモデル名 (ご指定の無料モデル)
model_name = "google/gemma-3-27b-it:free" 

# --- APIクライアントの初期化 ---

try:
    # OpenAIクライアントをOpenRouterのエンドポイントとキーで初期化
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url
    )
except Exception as e:
    print(f"クライアントの初期化中にエラーが発生しました: {e}")
    exit()

# --- API呼び出し関数 ---

def test_openrouter_api(prompt: str):
    """
    OpenRouter経由でGemma 3モデルにリクエストを送信し、応答を出力します。
    """
    print(f"--- APIリクエストを開始します (モデル: {model_name}) ---")
    print(f"プロンプト: '{prompt}'\n")

    # API呼び出し (OpenAIと同じ形式を使用)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # オプションのパラメータ
            max_tokens=500, # Gemma 3 (27B) は大規模なので、トークン数は少し多めに設定
            temperature=0.7
        )

        # 応答の表示
        if response.choices and response.choices[0].message:
            model_response = response.choices[0].message.content.strip()
            print("【Gemma 3 の応答】")
            print(model_response)
        else:
            print("【エラー】応答データが空か、期待される形式ではありませんでした。")

        # 使用トークン数の表示 (デバッグ情報)
        if response.usage:
            print(f"\n--- 使用情報 ---")
            print(f"総トークン数: {response.usage.total_tokens}")
            # 注意: 無料モデルの場合でも、使用量計測は行われます。

    except Exception as e:
        print(f"\n【API呼び出し中にエラーが発生しました】")
        print(f"エラー内容: {e}")
        # 原因: APIキーの誤り、無料枠のクォータ超過、またはモデル名の誤りなどが考えられます。

# --- メイン実行 ---

if __name__ == "__main__":
    # テスト用のプロンプト (物理研究/レゴプログラミングに関連した質問)
    test_prompt = input('なんて聞く？')

    # APIをテスト実行
    test_openrouter_api(test_prompt)
