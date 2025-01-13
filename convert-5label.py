import pandas as pd


def main():
    # Load dataset
    df = pd.read_csv("/home/ponydasierra/thuctap/news_classification/data/raw/news_dataset.csv")  # Thay "dataset.csv" bằng đường dẫn thực tế

    # Chuyển đổi các label dựa trên fake_ratio
    def convert_label(fake_ratio):
        if fake_ratio <= 20:
            return 'true'
        elif 20 < fake_ratio <= 40:
            return 'ptrue'
        elif 40 < fake_ratio <= 60:
            return 'pof'
        elif 60 < fake_ratio <= 80:
            return 'pfalse'
        else:
            return 'false'

    # Áp dụng chuyển đổi
    df['label'] = df['fake_ratio [%]'].apply(convert_label)

    # Lưu dataset mới
    df.to_csv("/home/ponydasierra/thuctap/news_classification/data/raw/dataset_converted5.csv", index=False)

if __name__ == "__main__":
    main()