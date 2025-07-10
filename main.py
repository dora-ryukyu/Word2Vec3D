import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sys

def visualize_word_vectors_rgb():
    """
    単語ベクトルを3D空間に可視化し、RGBカラーマッピングに基づいて色付けします。
    ユーザーは新しい単語を入力して、既存のプロットに追加できます。
    """

    # 1. モデルのロード
    # 使用するSentenceTransformerモデルの名前
    # 'cl-nagoya/ruri-v3-130m' は日本語に対応した軽量なモデルです。
    model_name = 'cl-nagoya/ruri-v3-130m'
    print(f"モデル '{model_name}' をロードしています...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        print("インターネット接続を確認するか、モデル名が正しいか確認してください。")
        return

    # 2. ベースとなる単語リストの定義
    # 可視化の基準となる単語カテゴリと単語のリスト
    word_categories = {
        "果物": ["りんご", "バナナ", "ぶどう", "いちご", "桃", "みかん", "梨", "柿", "さくらんぼ", "メロン"],
        "動物": ["犬", "猫", "ライオン", "象", "うさぎ", "パンダ", "キリン", "シマウマ", "ペンギン", "イルカ"],
        "乗り物": ["車", "電車", "飛行機", "船", "自転車", "新幹線", "ロケット", "バス", "トラック", "オートバイ"],
        "感情": ["嬉しい", "悲しい", "怒り", "楽しい", "驚き", "絶望", "興奮", "不安", "穏やか", "退屈"],
        "自然": ["山", "海", "空", "太陽", "月", "星", "雲", "雷", "川", "森"],
        "職業": ["医者", "教師", "エンジニア", "警察官", "消防士", "弁護士", "宇宙飛行士", "料理人", "芸術家", "農家"],
        "文房具": ["鉛筆", "消しゴム", "ノート", "定規", "ハサミ", "ペン", "のり", "絵の具", "クレヨン", "コンパス"],
        "抽象概念": ["愛", "平和", "自由", "時間", "夢", "知識", "正義", "真実", "幸福", "勇気"]
    }

    # 全ての単語とそれに対応するカテゴリをフラットなリストに格納
    all_words = []
    all_categories = []
    for category, words in word_categories.items():
        all_words.extend(words)
        all_categories.extend([category] * len(words))

    # 3. 単語のベクトル変換と次元削減
    print("単語をベクトルに変換し、3次元に削減しています...")
    # SentenceTransformerを使用して単語をベクトルにエンコード
    word_vectors = model.encode(all_words, show_progress_bar=True)
    # PCA (主成分分析) を使用して、高次元ベクトルを3次元に削減
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(word_vectors)

    # 4. 座標をRGBカラーに変換
    # MinMaxScalerを使用して、3次元座標を0-1の範囲に正規化
    # これにより、各成分をRGB値 (0-255) にマッピングできるようになります
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(vectors_3d)
    
    colors_rgb = []
    for vec in normalized_vectors:
        # 正規化された各成分を255倍してRGB値に変換
        r = int(vec[0] * 255)
        g = int(vec[1] * 255)
        b = int(vec[2] * 255)
        colors_rgb.append(f'rgb({r},{g},{b})')

    # 5. PlotlyのFigureオブジェクトを作成
    fig = go.Figure()

    # 各カテゴリの単語をプロットに追加
    for category_name in word_categories.keys():
        # 現在のカテゴリに属する単語のインデックスを取得
        indices = [i for i, cat in enumerate(all_categories) if cat == category_name]
        # 該当する単語のカラーを取得
        category_colors = [colors_rgb[i] for i in indices]

        # Scatter3dトレースを追加
        fig.add_trace(go.Scatter3d(
            x=vectors_3d[indices, 0],  # X座標 (主成分1)
            y=vectors_3d[indices, 1],  # Y座標 (主成分2)
            z=vectors_3d[indices, 2],  # Z座標 (主成分3)
            text=[all_words[i] for i in indices],  # 単語のテキストラベル
            mode='markers+text',  # マーカーとテキストの両方を表示
            textposition='top center',  # テキストの位置
            textfont=dict(size=10, color='#333'),  # テキストのフォントスタイル
            hoverinfo='text',  # ホバー時に単語テキストを表示
            marker=dict(
                symbol='circle',  # マーカーの形状
                color=category_colors,  # マーカーの色 (RGBマッピングされた色)
                size=6,  # マーカーのサイズ
                opacity=0.9,  # マーカーの不透明度
                line=dict(color='rgba(50, 50, 50, 0.5)', width=1)  # マーカーの境界線
            ),
            name=category_name  # 凡例に表示されるカテゴリ名
        ))
    
    # 6. プロットのレイアウト設定
    fig.update_layout(
        title=dict(
            text='<b>Word2Vec3D(単語ベクトルの3D可視化)</b>',  # プロットのタイトル
            y=0.95, x=0.5, xanchor='center', yanchor='top',
            font=dict(size=20, color='#222')
        ),
        scene=dict(
            xaxis=dict(title='主成分1 (Red)', backgroundcolor="rgba(0,0,0,0.02)", gridcolor="white", zerolinecolor="white"),
            yaxis=dict(title='主成分2 (Green)', backgroundcolor="rgba(0,0,0,0.02)", gridcolor="white", zerolinecolor="white"),
            zaxis=dict(title='主成分3 (Blue)', backgroundcolor="rgba(0,0,0,0.02)", gridcolor="white", zerolinecolor="white"),
            aspectmode='cube' # アスペクト比を立方体に設定し、軸のスケールを均一にする
        ),
        legend=dict(
            title='<b>カテゴリ</b>', bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='#ccc', borderwidth=1, itemsizing='constant'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest' # ホバーモードを設定
    )

    # プロットを表示
    fig.show()

    # 7. ユーザー入力による新しい単語の追加処理
    print("\n新しい単語を入力してください。(終了するには 'q' または 'exit' を入力)")
    while True:
        try:
            new_word = input("単語 > ").strip() # 入力の前後の空白を削除
            if new_word.lower() in ['q', 'exit']:
                print("プログラムを終了します。")
                break
            if not new_word: # 空の入力をスキップ
                continue

            # 新しい単語をベクトル化し、既存のPCAモデルで3次元に変換
            new_vector = model.encode([new_word])
            new_vector_3d = pca.transform(new_vector)
            
            # 入力単語の座標を正規化し、RGBカラーに変換
            # MinMaxScalerはfit_transformを一度実行済みなので、transformのみを使用
            normalized_new_vector = scaler.transform(new_vector_3d)
            vec = normalized_new_vector[0]
            r_new, g_new, b_new = int(vec[0] * 255), int(vec[1] * 255), int(vec[2] * 255)
            new_color = f'rgb({r_new},{g_new},{b_new})'
            
            # 新しい単語をプロットに追加
            fig.add_trace(go.Scatter3d(
                x=new_vector_3d[:, 0], y=new_vector_3d[:, 1], z=new_vector_3d[:, 2],
                mode='markers+text',
                text=[f"<b>{new_word}</b>"],
                textposition='bottom center',
                textfont=dict(size=14, color='black'),
                marker=dict(
                    symbol='circle',
                    size=12,
                    color=new_color,
                    line=dict(color='black', width=3) # 強調表示のための太い枠線
                ),
                name=f'入力: {new_word}' # 凡例に表示される名前
            ))
            
            print(f"'{new_word}' をプロットしました。")
            fig.show() # 更新されたプロットを表示

        except KeyboardInterrupt:
            print("\nプログラムを終了します。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("単語のエンコードまたはプロット中に問題が発生した可能性があります。")

if __name__ == '__main__':
    visualize_word_vectors_rgb()
