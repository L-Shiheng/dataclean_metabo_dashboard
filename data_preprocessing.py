import pandas as pd
import numpy as np
import re

def parse_metdna_file(file_buffer, file_type='csv'):
    """
    解析 MetDNA 导出文件，返回：
    1. df_transposed: 标准化后的数据表 (行=样本, 列=代谢物)
    2. feature_meta: 特征元数据表 (包含 Name, Confidence_Level 等，用于过滤和绘图)
    3. error_message: 错误信息
    """
    # 1. 读取文件
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 2. 识别元数据列 (MetDNA 标准列)
    metdna_meta_cols = [
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    ]
    
    existing_meta = [c for c in df.columns if c in metdna_meta_cols]
    sample_cols = [c for c in df.columns if c not in existing_meta]
    
    if not sample_cols:
        return None, None, "未找到样本数据列，请检查文件格式。"

    # 3. 构建唯一 ID 和元数据记录
    # 确保有 name 列
    if 'name' not in df.columns:
        df['name'] = np.nan
    if 'confidence_level' not in df.columns:
        df['confidence_level'] = 'Unknown'

    feature_meta_list = []
    
    def process_row(row):
        # 原始名字
        raw_name = str(row['name']) if pd.notna(row['name']) else ""
        raw_name = raw_name.strip()
        
        # 判断是否被注释 (有名字且不为空)
        is_annotated = (raw_name != "") and (raw_name.lower() != "nan")
        
        # 构建显示用的 ID
        if is_annotated:
            # 取第一个分号前的名字
            clean_name = raw_name.split(';')[0]
            # 如果有重名，后续处理
            unique_id = clean_name
        else:
            # 无注释，用 m/z 和 RT
            unique_id = f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}"
            
        return {
            "Metabolite_ID": unique_id,
            "Original_Name": raw_name,
            "Confidence_Level": row.get('confidence_level', 'Unknown'),
            "Is_Annotated": is_annotated
        }

    # 应用处理
    meta_info = df.apply(process_row, axis=1)
    meta_df = pd.DataFrame(meta_info.tolist())
    
    # 处理 ID 重名问题 (添加后缀 _1, _2)
    meta_df['Metabolite_ID'] = make_unique(meta_df['Metabolite_ID'])
    
    # 设置索引，方便查找
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 4. 构建转置后的数据表
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index # 使用处理后的唯一 ID 作为索引
    
    df_transposed = df_data.T
    
    # 5. 自动生成 Group 列
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    def guess_group(s):
        s_no_digit = re.sub(r'\d+$', '', str(s))
        s_clean = s_no_digit.rstrip('._-')
        return s_clean if s_clean else "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, meta_df, None

def make_unique(series):
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

# --- 原有的清洗管道 (保持不变) ---
def data_cleaning_pipeline(df, group_col, 
                           missing_thresh=0.5, 
                           impute_method='min', 
                           norm_method='None', 
                           log_transform=True,
                           scale_method='None'):
    # 1. 识别
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 2. 过滤
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    # 3. 填充
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

    # 4. 归一化
    if norm_method == 'Sum':
        row_sums = data_df.sum(axis=1)
        mean_sum = row_sums.mean()
        data_df = data_df.div(row_sums, axis=0) * mean_sum
    elif norm_method == 'Median':
        row_medians = data_df.median(axis=1)
        mean_median = row_medians.mean()
        data_df = data_df.div(row_medians, axis=0) * mean_median

    # 5. Log
    if log_transform:
        if (data_df < 0).any().any(): pass 
        else: data_df = np.log2(data_df + 1)

    # 6. Scaling
    if scale_method == 'Auto':
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    # 7. 清理
    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    
    final_cols = data_df.columns.tolist()
    processed_df = pd.concat([meta_df, data_df], axis=1)
    
    return processed_df, final_cols
