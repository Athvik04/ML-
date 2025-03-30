# learning source https://www.ketutrare.com/2018/11/algoritma-k-nearest-neighbor-dan-contoh-soal.html
import numpy as np
from knn import KNN

def main():
    # DATASET_DIR = 'knn-sheets.csv'
    # DATA_UJI_DIR = 'data-uji.csv'
    DATASET_DIR = 'd:/5th sem/Machiene learning/simple-knn-main/simple-knn-main/knn-sheets.csv'
    DATA_UJI_DIR = 'd:/5th sem/Machiene learning/simple-knn-main/simple-knn-main/data-uji.csv'

    k_value = 3
    knn = KNN(DATASET_DIR,DATA_UJI_DIR,k_value)
    result = knn.get_predict_result()
    print(f'result prediction : {result}')


if __name__ == '__main__':
    main()

