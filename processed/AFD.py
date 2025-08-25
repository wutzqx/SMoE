import numpy as np
from statsmodels.tsa.stattools import adfuller


# 对多元时间序列进行 ADF 检验
def adf_test_multivariate(time_series):
    results = {}
    adf = []
    for i in range(time_series.shape[1]):  # 对每个变量进行检验
        series = time_series[:, i]
        try:
            result = adfuller(series, autolag='AIC')  # 使用 AIC 自动选择滞后阶数
        except ValueError as e:
            continue
        results[f'Variable_{i + 1}'] = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
        adf.append(result[0])
        print(i)
    adf = sum(adf)/len(adf)
    return results, adf


# 主函数
def main():
    # 从 .npy 文件中读取多元时间序列数据
    dataset = 'SMAP'
    file_path = f'{dataset}/P-1_train.npy'  # 替换为你的文件路径
    time_series = np.load(file_path)

    # 进行 ADF 检验
    adf_results, adf = adf_test_multivariate(time_series)

    # 输出结果
    for var, result in adf_results.items():
        print(f"Results for {var}:")
        print(f"  ADF Statistic: {result['ADF Statistic']}")
        print(f"  p-value: {result['p-value']}")
        print("  Critical Values:")
        for key, value in result['Critical Values'].items():
            print(f"    {key}: {value}")
        print()
    print(f'ADF: {adf}')

# 运行主函数
if __name__ == "__main__":
    main()