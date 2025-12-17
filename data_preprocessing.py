import pandas as pd
import numpy as np
import re

def parse_metdna_file(file_buffer, file_type='csv', sample_start_col_index=28):
    """
    解析 MetDNA 导出文件 (从 AC 列开始读取数据)
    
    参数:
    sample_start_col_index: int, 默认 28 (对应 Excel 的 'AC' 列). 
                            0-based index, so A=0... Z=25, AA=26, AB=27, AC=28.
    """
    # 1. 读取文件
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 2. 强制分离元数据和样本数据
    # MetDNA 结构：前 28 列 (A-AB) 是注释信息，从 AC (索引28) 开始是样本
    if df.shape[1] <= sample_start_col_index:
        return None, None, f"文件列数不足！需要至少 {sample_start_col_index+1} 列 (至 AC 列)。"

    # 分割 DataFrame
    df_meta_cols = df.iloc[:, :sample_start_col_index] # A 到 AB
    df_samples = df.iloc[:, sample_start_col_index:]   # AC 到 最后
    
    # 3. 提取特征元数据 (Feature Metadata)
    # 我们需要找到 'name', 'mz', 'rt' 这些关键列用于构建 ID
    # 由于我们切分了 df，现在需要从 df_meta_cols 里找
    
    # 确保必要列存在 (大小写不敏感处理)
    cols_map = {c.lower(): c for c in df_meta_cols.columns}
    
    # 获取 name, mz, rt, confidence_level 的实际列名
    name_col = cols_map.get('name')
    mz_col = cols_map.get('mz')
    rt_col = cols_map.get('rt')
    conf_col = cols_map.get('confidence_level')

    if not (mz_col and rt_col):
        return None, None, "未在前 28 列中找到 'mz' 或 'rt' 列，请检查表头。"

    feature_meta_list = []
    
    def process_row(row):
        # 处理名字
        raw_name = str(row[name_col]) if (name_col and pd.notna(row[name_col])) else ""
        raw_name = raw_name.strip()
        
        # 判断是否被注释
        is_annotated = (raw_name != "") and (raw_name.lower() != "nan")
        
        # 构建 ID
        if is_annotated:
            clean_name = raw_name.split(';')[0]
            unique_id = clean_name
        else:
            # 无注释，用 m/z 和 RT 构建 ID
            mz_val = row[mz_col]
            rt_val = row[rt_col]
            unique_id = f"m/z{mz_val:.4f}_RT{rt_val:.2f}"
            
        return {
            "Metabolite_ID": unique_id,
            "Original_Name": raw_name,
            "Confidence_Level": row.get(conf_col, 'Unknown') if conf_col else 'Unknown',
            "Is_Annotated": is_annotated
        }

    # 生成元数据索引
    meta_info = df_meta_cols.apply(process_row, axis=1)
    meta_df = pd.DataFrame(meta_info.tolist())
    
    # 处理重名 (Metabolite_ID 必须唯一)
    meta_df['Metabolite_ID'] = make_unique(meta_df['Metabolite_ID'])
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 4. 构建转置后的数据表
    # 将样本数据部分的索引设为上面生成的唯一 ID
    df_samples.index = meta_df.index
    
    # 转置: 行=样本, 列=特征
    df_transposed = df_samples.T
    
    # 5. 生成 Group 列
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 自动猜测分组
    def guess_group(s):
        s_no_digit = re.sub(r'\d+$', '', str(s))
        s_clean = s_no_digit.rstrip('._-')
        return s_clean if s_clean else "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, meta_df, None

def merge_multiple_metdna_files(uploaded_files):
    """
    处理多文件上传：
    1. 解析每个文件
    2. 按 SampleID 合并数据
    3. 去除重复的代谢物 (列名重复时，保留第一个文件的)
    """
    merged_df = None
    merged_meta = None
    all_errors = []

    for file in uploaded_files:
        # 重置文件指针
        file.seek(0)
        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
        
        # 解析
        df, meta, err = parse_metdna_file(file, file_type=file_type, sample_start_col_index=28)
        
        if err:
            all_errors.append(f"{file.name}: {err}")
            continue
            
        if merged_df is None:
            # 第一个文件，直接初始化
            merged_df = df
            merged_meta = meta
        else:
            # 后续文件，执行合并
            # 1. 元数据合并 (垂直堆叠)
            merged_meta = pd.concat([merged_meta, meta])
            
            # 2. 数据合并 (水平合并，基于 SampleID 和 Group)
            # 使用 outer merge 保证所有样本都被保留
            # suffixes 用于处理重名列，但我们稍后会手动去重
            merged_df = pd.merge(merged_df, df, on=['SampleID', 'Group'], how='outer', suffixes=('', '_dup_remove'))

    if merged_df is None:
        return None, None, "\n".join(all_errors)

    # --- 去重逻辑 ---
    # 1. 去除元数据中的重复索引 (特征名重复)
    merged_meta = merged_meta[~merged_meta.index.duplicated(keep='first')]
    
    # 2. 去除数据表中的重复列
    # 如果 merge 产生了 _dup_remove 后缀的列，说明是重复特征，直接删掉
    cols_to_remove = [c for c in merged_df.columns if c.endswith('_dup_remove')]
    if cols_to_remove:
        merged_df.drop(columns=cols_to_remove, inplace=True)
    
    # 3. 再次检查是否有重复列名 (双重保险)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    return merged_df, merged_meta, None

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
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 过滤
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    # 填充
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

    # 归一化
    if norm_method == 'Sum':
        row_sums = data_df.sum(axis=1)
        mean_sum = row_sums.mean()
        data_df = data_df.div(row_sums, axis=0) * mean_sum
    elif norm_method == 'Median':
        row_medians = data_df.median(axis=1)
        mean_median = row_medians.mean()
        data_df = data_df.div(row_medians, axis=0) * mean_median

    # Log
    if log_transform:
        if (data_df < 0).any().any(): pass 
        else: data_df = np.log2(data_df + 1)

    # Scaling
    if scale_method == 'Auto':
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    # 清理
    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    
    final_cols = data_df.columns.tolist()
    processed_df = pd.concat([meta_df, data_df], axis=1)
    
    return processed_df, final_cols
