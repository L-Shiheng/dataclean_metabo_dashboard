import pandas as pd
import numpy as np
import re
import os

def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    """
    解析 MetDNA 导出文件 (v3.2 智能去重版)
    """
    # 1. 读取文件
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 2. 智能识别样本列
    known_meta_cols = [
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    ]
    
    sample_cols = []
    
    for col in df.columns:
        if col in known_meta_cols: continue
        try:
            # 简单判断：如果是数值列且大部分非空，认为是样本
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.notna().mean() > 0.5:
                sample_cols.append(col)
        except: pass
            
    if not sample_cols:
        return None, None, "未找到样本数据列。"

    # 3. 构建元数据
    # 使用文件名做标签，防止ID重复
    file_tag = os.path.splitext(os.path.basename(file_name))[0]
    file_tag = re.sub(r'[^a-zA-Z0-9]', '_', file_tag)
    
    if 'name' not in df.columns: df['name'] = np.nan
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'

    def process_row(row):
        raw_name = str(row['name']).strip() if pd.notna(row['name']) else ""
        is_annotated = (raw_name != "") and (raw_name.lower() != "nan")
        
        if is_annotated:
            clean_name = raw_name.split(';')[0] # 纯净名字，用于跨文件比对
            # ID 必须包含 file_tag 以区分来源
            unique_id = f"{clean_name}_{file_tag}"
        else:
            # 未注释的用 m/z RT
            unique_id = f"m/z{row['mz']:.4f}_RT{row['rt']:.2f}_{file_tag}"
            clean_name = unique_id # 未注释的“名字”就是它的ID，确保独一无二
            
        return {
            "Metabolite_ID": unique_id,
            "Original_Name": raw_name,
            "Clean_Name": clean_name, # 这是“去重”的关键依据
            "Confidence_Level": row.get('confidence_level', 'Unknown'),
            "Is_Annotated": is_annotated,
            "Source_File": file_tag
        }

    meta_info = df.apply(process_row, axis=1)
    meta_df = pd.DataFrame(meta_info.tolist())
    
    # 确保单文件内 ID 唯一
    meta_df['Metabolite_ID'] = make_unique(meta_df['Metabolite_ID'])
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 4. 提取数据
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    
    # 5. 生成 SampleID 和 Group
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    def guess_group(s):
        s_no = re.sub(r'\d+$', '', str(s))
        return s_no.rstrip('._-') or "Unknown"

    df_transposed.insert(1, 'Group', df_transposed['SampleID'].apply(guess_group))
    
    return df_transposed, meta_df, None

def merge_multiple_dfs(results_list):
    """
    合并多文件逻辑 (v2.0): 
    同名代谢物只保留峰面积最大者 (Max Peak Area Selection)
    """
    if not results_list: return None, None, "无数据"
    
    # --- 步骤 1: 竞选“最佳代谢物” ---
    # 字典结构: { Clean_Name: (File_Index, Feature_ID, Max_Intensity) }
    best_features = {}
    
    for file_idx, (df, meta, fname) in enumerate(results_list):
        # 计算该文件中每个特征的总丰度 (Sum Intensity)
        # 先排除非数值列 (SampleID, Group)
        numeric_df = df.select_dtypes(include=[np.number])
        intensities = numeric_df.sum(axis=0) # 对列求和
        
        for feat_id in numeric_df.columns:
            # 获取该特征的“纯净名”
            try:
                clean_name = meta.loc[feat_id, 'Clean_Name']
                is_annotated = meta.loc[feat_id, 'Is_Annotated']
            except KeyError:
                continue # 异常情况跳过
            
            # 获取当前强度
            curr_score = intensities.get(feat_id, 0)
            
            # 核心逻辑：
            # 如果是未注释特征，因为它包含了 RT 和 m/z，通常视作唯一，直接保留(通过将 ID 作为 key)
            # 如果是已注释特征，使用 Clean_Name 作为 key 进行竞争
            
            # 如果尚未记录该代谢物，直接存入
            if clean_name not in best_features:
                best_features[clean_name] = (file_idx, feat_id, curr_score)
            else:
                # 如果已存在，比较强度
                prev_idx, prev_id, prev_score = best_features[clean_name]
                if curr_score > prev_score:
                    # 当前文件中的这个代谢物更强，替换掉之前的
                    best_features[clean_name] = (file_idx, feat_id, curr_score)
    
    # --- 步骤 2: 构建过滤后的数据表 ---
    
    # 按文件索引整理需要保留的特征 ID
    # 结构: { file_index: [feat_id_1, feat_id_2...] }
    files_features_to_keep = {i: [] for i in range(len(results_list))}
    
    for c_name, (f_idx, f_id, score) in best_features.items():
        files_features_to_keep[f_idx].append(f_id)
        
    dfs_to_concat = []
    base_group_series = None
    
    for i, (df, meta, fname) in enumerate(results_list):
        # 准备数据：设 SampleID 为索引
        if 'SampleID' in df.columns:
            df = df.set_index('SampleID')
            
        # 提取 Group (只取第一个非空文件的)
        if 'Group' in df.columns:
            if base_group_series is None:
                base_group_series = df['Group']
            df = df.drop(columns=['Group'])
            
        # 关键步骤：只保留竞选成功的特征列
        cols_to_keep = files_features_to_keep[i]
        # 确保这些列真的在 df 里 (双重保险)
        valid_cols = [c for c in cols_to_keep if c in df.columns]
        
        df_filtered = df[valid_cols]
        dfs_to_concat.append(df_filtered)
        
    # --- 步骤 3: 拼接 (Concat) ---
    try:
        full_df = pd.concat(dfs_to_concat, axis=1, join='outer')
    except Exception as e:
        return None, None, f"合并出错: {str(e)}"
    
    full_df.fillna(0, inplace=True)
    
    # 恢复 Group
    if base_group_series is not None:
        aligned_group = base_group_series.reindex(full_df.index).fillna('Unknown')
        full_df.insert(0, 'Group', aligned_group)
    else:
        full_df.insert(0, 'Group', 'Unknown')
        
    full_df.reset_index(inplace=True)
    full_df.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # --- 步骤 4: 整理元数据 ---
    # 找出最终保留的所有 Feature ID
    final_ids = [fid for f_list in files_features_to_keep.values() for fid in f_list]
    
    # 把所有文件的元数据拼起来，然后只保留最终留下的那些
    all_meta = pd.concat([res[1] for res in results_list])
    merged_meta = all_meta.loc[final_ids]
    
    return full_df, merged_meta, None

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

# --- 清洗管道 (不变) ---
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
