# 文件名: data_preprocessing.py
import pandas as pd
import numpy as np
import re

def parse_metdna_file(file_buffer, file_type='csv'):
    """
    专门解析 MetDNA 导出的特征表
    """
    # 1. 读取文件
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, f"读取失败: {str(e)}"

    # 2. 识别关键列
    # MetDNA 导出的标准注释列 (我们要排除这些，剩下的就是样本)
    # 根据您提供的文件，这些是常见的元数据列
    metdna_meta_cols = [
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    ]
    
    # 找出实际存在的元数据列
    existing_meta = [c for c in df.columns if c in metdna_meta_cols]
    
    # 剩下的列我们就认为是样本列
    sample_cols = [c for c in df.columns if c not in existing_meta]
    
    if not sample_cols:
        return None, "未找到样本数据列，请检查文件格式。"

    # 3. 构建代谢物唯一名称 (处理无注释的情况)
    # 如果有 'name' 列且不为空，用 name；否则用 mz_rt
    if 'name' in df.columns:
        # 填充空值
        df['name'] = df['name'].fillna('')
        
        def get_name(row):
            # 如果名字为空，或者看起来像无意义字符
            if not row['name'] or str(row['name']).strip() == '':
                # 使用 m/z 和 RT 作为代号
                return f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}"
            return str(row['name']).split(';')[0] # 有些名字有多个分号，取第一个
            
        df['Metabolite_ID'] = df.apply(get_name, axis=1)
    else:
        # 如果连 name 列都没有
        df['Metabolite_ID'] = df.apply(lambda row: f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}", axis=1)

    # 确保列名唯一 (防止重名代谢物)
    df['Metabolite_ID'] = make_unique(df['Metabolite_ID'])

    # 4. 提取数据并转置
    # 我们只需要 Sample 列，索引变成 Metabolite_ID
    df_data = df[sample_cols].copy()
    df_data.index = df['Metabolite_ID']
    
    # 转置: 行变成样本，列变成代谢物
    df_transposed = df_data.T
    
    # 5. 自动生成分组列 (Group)
    # 将索引(样本名)变成一列 SampleID
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 尝试从样本名推断分组 (简单启发式：去掉末尾的数字)
    # 例如: DK.BT.1208.HT1 -> DK.BT.1208.HT
    def guess_group(s):
        # 去掉末尾的数字
        s_no_digit = re.sub(r'\d+$', '', str(s))
        # 去掉末尾可能残留的分隔符
        s_clean = s_no_digit.rstrip('._-')
        # 如果处理后变成空了(例如样本名叫 "1", "2")，就退回到原名
        return s_clean if s_clean else "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, None

def make_unique(series):
    """解决重名问题，给重复项加后缀 _1, _2"""
    seen = set()
    result = []
    for item in series:
        new_item = item
        counter = 1
        while new_item in seen:
            new_item = f"{item}_{counter}"
            counter += 1
        seen.add(new_item)
        result.append(new_item)
    return result

# --- 原有的数据清洗管道 (保持不变) ---
def data_cleaning_pipeline(df, group_col, 
                           missing_thresh=0.5, 
                           impute_method='min', 
                           norm_method='None', 
                           log_transform=True,
                           scale_method='None'):
    """
    标准代谢组学数据清洗流程
    """
    # 1. 自动识别数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 2. 缺失值过滤
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    # 3. 缺失值填充
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min':
            min_vals = data_df.min() * 0.5
            data_df = data_df.fillna(min_vals)
        elif impute_method == 'mean':
            data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'median':
            data_df = data_df.fillna(data_df.median())
        elif impute_method == 'zero':
            data_df = data_df.fillna(0)
        data_df = data_df.fillna(0)

    # 4. 样本归一化
    if norm_method == 'Sum':
        row_sums = data_df.sum(axis=1)
        mean_sum = row_sums.mean()
        data_df = data_df.div(row_sums, axis=0) * mean_sum
    elif norm_method == 'Median':
        row_medians = data_df.median(axis=1)
        mean_median = row_medians.mean()
        data_df = data_df.div(row_medians, axis=0) * mean_median

    # 5. 数据转换
    if log_transform:
        if (data_df < 0).any().any():
            pass # 含有负数不Log
        else:
            data_df = np.log2(data_df + 1)

    # 6. 数据缩放
    if scale_method == 'Auto':
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    # 7. 清理低方差
    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    
    final_cols = data_df.columns.tolist()
    processed_df = pd.concat([meta_df, data_df], axis=1)
    
    return processed_df, final_cols
