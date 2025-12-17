import pandas as pd
import numpy as np
import re
import os

def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    """
    解析 MetDNA 导出文件 (升级版 v2.0)
    1. 自动检测样本列 (基于数值类型，不再依赖固定列位置)
    2. 返回转置后的数据和元数据
    """
    # 1. 读取文件
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 2. 智能识别样本列 (核心修改)
    # MetDNA 的元数据列通常包含文字或混合内容，而样本列全是数字
    # 策略：尝试将每一列转换为数字，如果成功且非空值比例高，则是样本列
    
    # 已知的必须排除的元数据列名 (白名单)
    known_meta_cols = [
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    ]
    
    sample_cols = []
    meta_cols = []
    
    for col in df.columns:
        if col in known_meta_cols:
            meta_cols.append(col)
            continue
            
        # 尝试转换为数值
        try:
            # coerce会将无法转换的变成NaN
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # 如果非NaN的数据超过 50%，我们认为它是样本数据 (MetDNA 样本列通常是丰度值)
            if numeric_series.notna().mean() > 0.5:
                sample_cols.append(col)
            else:
                meta_cols.append(col)
        except:
            meta_cols.append(col)
            
    if not sample_cols:
        return None, None, "未找到有效的样本数据列 (数值列)。"

    # 3. 构建唯一 ID (加入文件名后缀以防多文件合并时重名)
    # 提取简单的文件名标识 (如 'Pos', 'Neg')
    file_tag = os.path.splitext(file_name)[0]
    # 清理文件名，只保留最后一段有意义的，防止名字太长
    if '_' in file_tag: file_tag = file_tag.split('_')[-1]
    
    if 'name' not in df.columns: df['name'] = np.nan
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'

    meta_list = []
    
    def process_row(row):
        raw_name = str(row['name']).strip() if pd.notna(row['name']) else ""
        is_annotated = (raw_name != "") and (raw_name.lower() != "nan")
        
        if is_annotated:
            clean_name = raw_name.split(';')[0]
            # 关键：在名字后面加上文件标识，防止不同模式下的同名化合物混淆
            # 例如: Glucose (Pos) vs Glucose (Neg)
            unique_id = f"{clean_name}_{file_tag}"
        else:
            unique_id = f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}_{file_tag}"
            
        return {
            "Metabolite_ID": unique_id,
            "Original_Name": raw_name,
            "Clean_Name": clean_name if is_annotated else unique_id,
            "Confidence_Level": row.get('confidence_level', 'Unknown'),
            "Is_Annotated": is_annotated,
            "Source_File": file_tag
        }

    meta_info = df.apply(process_row, axis=1)
    meta_df = pd.DataFrame(meta_info.tolist())
    
    # 再次确保 ID 唯一 (防止同个文件内就有重名)
    meta_df['Metabolite_ID'] = make_unique(meta_df['Metabolite_ID'])
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 4. 构建数据表
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    
    # 5. 生成 SampleID 和 Group
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 统一 Group 推断逻辑
    def guess_group(s):
        s_no_digit = re.sub(r'\d+$', '', str(s))
        s_clean = s_no_digit.rstrip('._-')
        return s_clean if s_clean else "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, meta_df, None

def merge_multiple_dfs(results_list):
    """
    合并多个已解析的数据表
    results_list: list of (df, meta, filename)
    """
    if not results_list: return None, None, "无数据"
    
    # 1. 合并元数据 (Feature Meta)
    # 直接上下拼接所有文件的元数据
    all_meta_dfs = [res[1] for res in results_list]
    merged_meta = pd.concat(all_meta_dfs)
    
    # 2. 合并数据表 (Data Frames)
    # 我们需要按 SampleID 进行合并 (横向拼接特征)
    # 基准表: 取第一个文件
    base_df = results_list[0][0]
    
    for i in range(1, len(results_list)):
        next_df = results_list[i][0]
        
        # 检查样本ID是否一致
        # 我们使用 'SampleID' 作为连接键
        # 注意：如果不同文件里样本顺序不一样，merge 会自动对齐
        # 但是，必须保证 SampleID 的命名是一模一样的
        
        # 为了安全，先移除 Group 列 (避免重复 Group_x, Group_y)，合并后再加回来
        cols_to_use = [c for c in next_df.columns if c != 'Group']
        
        base_df = pd.merge(base_df, next_df[cols_to_use], on='SampleID', how='outer')
        
    # 重新整理 Group 列 (如果 outer join 导致某些样本 Group 丢失，尝试恢复)
    # 这里假设第一个文件的 Group 是准的
    if 'Group' not in base_df.columns:
        # 应该不会发生，因为 base_df 保留了 Group
        pass
        
    # 填充因为 outer join 可能产生的 NaN (某个样本在某个文件里没有测到)
    # 在代谢组学合并中，如果缺失通常意味着未检出，填 0
    base_df.fillna(0, inplace=True)
        
    return base_df, merged_meta, None

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

# --- 以下为原有的数据清洗管道 (无需变动) ---
def data_cleaning_pipeline(df, group_col, missing_thresh=0.5, impute_method='min', 
                           norm_method='None', log_transform=True, scale_method='None'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min': data_df = data_df.fillna(data_df.min() * 0.5)
        elif impute_method == 'mean': data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'median': data_df = data_df.fillna(data_df.median())
        elif impute_method == 'zero': data_df = data_df.fillna(0)
        data_df = data_df.fillna(0)

    if norm_method == 'Sum':
        data_df = data_df.div(data_df.sum(axis=1), axis=0) * data_df.sum(axis=1).mean()
    elif norm_method == 'Median':
        data_df = data_df.div(data_df.median(axis=1), axis=0) * data_df.median(axis=1).mean()

    if log_transform:
        if not (data_df < 0).any().any(): data_df = np.log2(data_df + 1)

    if scale_method == 'Auto':
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    
    return pd.concat([meta_df, data_df], axis=1), data_df.columns.tolist()
