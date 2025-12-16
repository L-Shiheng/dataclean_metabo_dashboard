import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.statimport streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 导入数据清洗模块
# ==========================================
try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file
except ImportError:
    st.error("❌ 严重错误：未找到 'data_preprocessing.py'。请确保该文件在同一目录下。")
    st.stop()

# ==========================================
# 1. 全局配置与样式
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {
        font-size: 16px; font-weight: bold; padding: 10px 15px;
        background-color: white; border-radius: 5px 5px 0 0;
    }
    .stMultiSelect [data-baseweb="tag"] {background-color: #e3e8ee;}
</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# --- 通用绘图布局 ---
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white",
        width=width, height=height,
        title={
            'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color='black', family="Arial, bold")
        },
        xaxis=dict(title=x_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold")),
        yaxis=dict(title=y_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold"), automargin=True),
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.15, bordercolor="Black", borderwidth=0, font=dict(size=12)),
        margin=dict(l=80, r=180, t=80, b=80)
    )
    return fig

# PLS-DA 椭圆
def get_ellipse_coordinates(x, y, std_mult=2):
    if len(x) < 3: return None, None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * std_mult * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, 100)
    ell_x = width/2 * np.cos(t)
    ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

def calculate_vips(model):
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features):
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        fc = np.mean(v1) - np.mean(v2) 
        try: t, p = stats.ttest_ind(v1, v2, equal_var=False)
        except: p = 1.0
        if np.isnan(p): p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty:
        _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = p_corr
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else:
        res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# ==========================================
# 2. 侧边栏与数据加载
# ==========================================
with st.sidebar:
    st.header("🛠️ 分析控制台")
    uploaded_file = st.file_uploader("1. 上传数据 (MetDNA / CSV)", type=["csv", "xlsx"])
    
    # 初始化变量
    feature_meta = None 

    if not uploaded_file:
        st.info("👋 请先上传数据")
        st.stop()
        
    # --- 智能文件解析 ---
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"无法读取文件: {e}"); st.stop()

    metdna_markers = ['peak_name', 'mz', 'rt', 'name', 'formula', 'confidence_level']
    is_metdna = any(col in raw_df.columns for col in metdna_markers)
    
    if is_metdna:
        st.success("✅ MetDNA 格式识别成功")
        uploaded_file.seek(0)
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
        # 调用新的解析函数，获取数据和特征元数据
        parsed_df, meta_df, err = parse_metdna_file(uploaded_file, file_type=file_type)
        if err: st.error(err); st.stop()
        
        raw_df = parsed_df
        feature_meta = meta_df # 保存元数据供后续使用
        
    
    # --- 流程控制 ---
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not non_num: st.error("❌ 无法识别分组列"); st.stop()
    
    default_grp_idx = non_num.index('Group') if 'Group' in non_num else 0
    group_col = st.selectbox("2. 分组列", non_num, index=default_grp_idx)

    # --- 新增：特征过滤选择 ---
    st.divider()
    st.markdown("### 3. 特征过滤")
    filter_option = st.radio("选择参与分析的特征:", ["全部特征", "仅已注释特征"], index=0, 
                             help="选择'仅已注释特征'将过滤掉没有化合物名称 (只有 m/z 和 RT) 的特征。")

    # --- 数据清洗 ---
    with st.expander("⚙️ 数据清洗 (高级)", expanded=False):
        miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5, 0.1)
        impute_m = st.selectbox("填充方法", ["min", "mean", "zero"], index=0)
        norm_m = st.selectbox("样本归一化", ["None", "Sum", "Median"], index=0)
        do_log = st.checkbox("Log2 转化", value=True)
        scale_m = st.selectbox("特征缩放", ["None", "Auto", "Pareto"], index=0)

    # --- 组别 ---
    all_groups = sorted(raw_df[group_col].astype(str).unique())
    st.divider()
    st.markdown("### 4. 组别与对比")
    selected_groups = st.multiselect("纳入分析的组 (全局):", all_groups, default=all_groups[:2] if len(all_groups)>=2 else all_groups)
    if len(selected_groups) < 2: st.error("至少选 2 个组"); st.stop()
    
    c1, c2 = st.columns(2)
    valid_groups = [g for g in selected_groups]
    case_grp = c1.selectbox("Exp (Case)", valid_groups, index=0)
    ctrl_grp = c2.selectbox("Ctrl (Ref)", valid_groups, index=1 if len(valid_groups)>1 else 0)
    
    st.divider()
    st.subheader("5. 绘图参数")
    p_th = st.number_input("P-value 阈值", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC 阈值", 1.0)
    enable_jitter = st.checkbox("火山图抖动 (Jitter)", value=True)

# ==========================================
# 3. 数据处理 Pipeline
# ==========================================

# A. 基础清洗
df_proc, feats = data_cleaning_pipeline(
    raw_df, group_col, missing_thresh=miss_th, impute_method=impute_m, 
    norm_method=norm_m, log_transform=do_log, scale_method=scale_m
)

# B. 应用特征过滤 (新功能)
if filter_option == "仅已注释特征":
    if feature_meta is not None:
        # 从元数据中找出 Is_Annotated 为 True 的 ID
        annotated_feats = feature_meta[feature_meta['Is_Annotated'] == True].index.tolist()
        # 取交集 (确保这些特征在清洗后还存在)
        feats = [f for f in feats if f in annotated_feats]
        if not feats:
            st.error("过滤后没有剩余特征！请检查数据或切换回 '全部特征'。")
            st.stop()
        st.success(f"已过滤：保留 {len(feats)} 个已注释特征")
    else:
        st.warning("当前非 MetDNA 格式数据，无法自动判断注释状态，已使用全部特征。")

df_sub = df_proc[df_proc[group_col].isin(selected_groups)].copy()

# C. 统计分析
if case_grp != ctrl_grp:
    res_stats = run_pairwise_statistics(df_sub, group_col, case_grp, ctrl_grp, feats)
    
    # --- 新增：将 Confidence Level 合并到统计结果中 ---
    if feature_meta is not None:
        # 合并元数据 (Left join on Metabolite name)
        res_stats = res_stats.merge(feature_meta[['Confidence_Level']], 
                                    left_on='Metabolite', right_index=True, how='left')
        # 填充缺失值
        res_stats['Confidence_Level'] = res_stats['Confidence_Level'].fillna('Unknown')
    else:
        res_stats['Confidence_Level'] = 'N/A'

    res_stats['Sig'] = 'NS'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
    sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()
else:
    res_stats = pd.DataFrame(); sig_metabolites = []

# ==========================================
# 4. 结果展示
# ==========================================
st.title("📊 代谢组学分析报告")
st.caption(f"对比: {case_grp} vs {ctrl_grp} | 分析特征数: {len(feats)} | 显著差异: {len(sig_metabolites)}")

tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP 特征", "🌋 火山图", "🔥 热图", "📑 详情"])

# --- Tab 1: PCA ---
with tabs[0]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X = StandardScaler().fit_transform(df_sub[feats])
            pca = PCA(n_components=2).fit(X)
            pcs = pca.transform(X)
            var = pca.explained_variance_ratio_
            fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col],
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
            st.plotly_chart(fig_pca, use_container_width=False)

# --- Tab 2: PLS-DA ---
with tabs[1]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X_pls = StandardScaler().fit_transform(df_sub[feats])
            y_labels = pd.factorize(df_sub[group_col])[0]
            pls_model = PLSRegression(n_components=2).fit(X_pls, y_labels)
            pls_scores = pls_model.x_scores_
            plot_df = pd.DataFrame({'C1': pls_scores[:,0], 'C2': pls_scores[:,1], 'Group': df_sub[group_col].values})
            fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            for i, grp in enumerate(selected_groups):
                sub_g = plot_df[plot_df['Group'] == grp]
                if len(sub_g) >= 3:
                    ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
                    if ell_x is not None:
                        color = GROUP_COLORS[i % len(GROUP_COLORS)]
                        fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', line=dict(color=color, width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
            fig_pls.update_traces(marker=dict(size=14, line=dict(width=1.5, color='black'), opacity=1.0))
            update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
            st.plotly_chart(fig_pls, use_container_width=False)

# --- Tab 3: VIP ---
with tabs[2]:
    st.markdown("### Top 25 VIP Features")
    if 'pls_model' in locals():
        vip_scores = calculate_vips(pls_model)
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
        top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
        
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            fig_vip = px.bar(top_vip, x="VIP", y="Metabolite", orientation='h',
                             color="VIP", color_continuous_scale="RdBu_r", width=800, height=700)
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_vip.update_traces(marker_line_color='black', marker_line_width=1.0)
            fig_vip.update_layout(
                template="simple_white", width=800, height=700,
                title={'text': "VIP Scores", 'x':0.5, 'xanchor': 'center', 'font': dict(size=20, family="Arial, bold")},
                xaxis=dict(title="VIP Score", showline=True, mirror=True, linewidth=2, linecolor='black'),
                yaxis=dict(title="", showline=True, mirror=True, linewidth=2, linecolor='black'),
                coloraxis_showscale=False,
                margin=dict(l=200, r=40, t=60, b=60) 
            )
            st.plotly_chart(fig_vip, use_container_width=False)

# --- Tab 4: 火山图 ---
with tabs[3]:
    if case_grp == ctrl_grp: st.warning("请选择不同的组")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            plot_df = res_stats.copy()
            x_c, y_c = "Log2_FC", "-Log10_P"
            if enable_jitter:
                np.random.seed(42)
                xr, yr = (plot_df[x_c].max()-plot_df[x_c].min()) or 1, (plot_df[y_c].max()-plot_df[y_c].min()) or 1
                plot_df['Log2_FC_J'] = plot_df[x_c] + np.random.normal(0, xr*0.015, len(plot_df))
                plot_df['-Log10_P_J'] = plot_df[y_c] + np.random.normal(0, yr*0.015, len(plot_df))
                x_c, y_c = "Log2_FC_J", "-Log10_P_J"
            
            # 准备 Hover 数据 (包含 Confidence Level)
            hover_dict = {"Metabolite":True, "Log2_FC":':.2f', "P_Value":':.2e', 
                          "Confidence_Level":True, # 新增
                          x_c:False, y_c:False}

            fig_vol = px.scatter(plot_df, x=x_c, y=y_c, color="Sig", color_discrete_map=COLOR_PALETTE,
                                 hover_data=hover_dict, # 应用新的 hover
                                 width=600, height=600)
            
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.update_traces(marker=dict(size=10, opacity=0.75, line=dict(width=1, color='black')))
            update_layout_square(fig_vol, f"Volcano: {case_grp} vs {ctrl_grp}", "Log2 Fold Change", "-Log10(P-value)")
            st.plotly_chart(fig_vol, use_container_width=False)

# --- Tab 5: 热图 ---
with tabs[4]:
    if not sig_metabolites: st.info("无显著差异物")
    else:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            top_n = 50
            top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
            hm_data = df_sub.set_index(group_col)[top_feats]
            lut = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(df_sub[group_col].unique())}
            row_colors = df_sub[group_col].map(lut)
            try:
                g = sns.clustermap(hm_data.astype(float), z_score=1, cmap="vlag", center=0, 
                                   row_colors=row_colors, figsize=(12, 12), 
                                   dendrogram_ratio=(.15, .15), 
                                   cbar_pos=(0.3, 1.02, 0.4, 0.03), cbar_kws={'orientation': 'horizontal'})
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", fontsize=10)
                g.ax_heatmap.set_yticklabels([]); g.ax_heatmap.set_ylabel("Samples", fontsize=12)
                st.pyplot(g.fig)
            except Exception as e: st.error(f"绘图错误: {e}")

# --- Tab 6: 详情 & 箱线图 ---
with tabs[5]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("统计表")
        if not res_stats.empty:
            display_df = res_stats.sort_values("P_Value").copy()
            # 格式化显示，加入 Confidence Level
            cols_to_show = ["Metabolite", "Log2_FC", "P_Value", "FDR", "Confidence_Level"]
            # 确保列存在
            cols_to_show = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(display_df[cols_to_show].style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"})
                         .background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
                         use_container_width=True, height=600)
    with c2:
        st.subheader("箱线图")
        feat_options = sorted(feats)
        def_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0
        target_feat = st.selectbox("选择代谢物", feat_options, index=def_ix)
        if target_feat:
            box_df = df_sub[[group_col, target_feat]].copy()
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             color_discrete_sequence=GROUP_COLORS, points="all", width=500, height=500)
            fig_box.update_traces(width=0.6, marker=dict(size=7, opacity=0.6, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
            update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
            st.plotly_chart(fig_box, use_container_width=False)
s.multitest import multipletests

# ==========================================
# 0. 导入数据清洗模块
# ==========================================
try:
    from data_preprocessing import data_cleaning_pipeline, parse_metdna_file
except ImportError:
    st.error("❌ 严重错误：未找到 'data_preprocessing.py'。请确保该文件在同一目录下。")
    st.stop()

# ==========================================
# 1. 全局配置与样式
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {
        font-size: 16px; font-weight: bold; padding: 10px 15px;
        background-color: white; border-radius: 5px 5px 0 0;
    }
    .stMultiSelect [data-baseweb="tag"] {background-color: #e3e8ee;}
</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# --- 通用绘图布局 ---
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white",
        width=width, height=height,
        title={
            'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color='black', family="Arial, bold")
        },
        xaxis=dict(title=x_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold")),
        yaxis=dict(title=y_title, showline=True, linewidth=2, linecolor='black', mirror=True, title_font=dict(size=16, family="Arial, bold"), automargin=True),
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.15, bordercolor="Black", borderwidth=0, font=dict(size=12)),
        margin=dict(l=80, r=180, t=80, b=80)
    )
    return fig

# PLS-DA 椭圆
def get_ellipse_coordinates(x, y, std_mult=2):
    if len(x) < 3: return None, None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * std_mult * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, 100)
    ell_x = width/2 * np.cos(t)
    ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

def calculate_vips(model):
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features):
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        fc = np.mean(v1) - np.mean(v2) 
        try: t, p = stats.ttest_ind(v1, v2, equal_var=False)
        except: p = 1.0
        if np.isnan(p): p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    res_df = pd.DataFrame(res).dropna()
    if not res_df.empty:
        _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = p_corr
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else:
        res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# ==========================================
# 2. 侧边栏与数据加载逻辑
# ==========================================
with st.sidebar:
    st.header("🛠️ 分析控制台")
    uploaded_file = st.file_uploader("1. 上传数据 (支持 CSV/Excel)", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.info("👋 请先上传数据 (MetDNA 导出文件或标准格式表)")
        st.stop()
        
    # --- 智能文件解析 ---
    # 先尝试作为标准 CSV 读取
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"无法读取文件: {e}")
        st.stop()

    # 检查是否为 MetDNA 格式 (通过检查关键列名)
    metdna_markers = ['peak_name', 'mz', 'rt', 'name', 'formula']
    is_metdna = any(col in raw_df.columns for col in metdna_markers)
    
    if is_metdna:
        st.success("✅ 检测到 MetDNA 格式，正在自动转换...")
        # 重新指针回到文件开头，因为上面 read 了一次
        uploaded_file.seek(0)
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
        
        parsed_df, err = parse_metdna_file(uploaded_file, file_type=file_type)
        if err:
            st.error(err); st.stop()
        
        raw_df = parsed_df
        st.markdown(f"**自动提取组别**: {', '.join(raw_df['Group'].unique())}")
        # MetDNA 解析后会自动生成 'Group' 列
    
    # --- 继续正常的分析流程 ---
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not non_num: st.error("❌ 无法识别分组列。"); st.stop()
    
    # 如果是 MetDNA 自动生成的 'Group'，默认选中它
    default_grp_idx = non_num.index('Group') if 'Group' in non_num else 0
    group_col = st.selectbox("2. 分组列", non_num, index=default_grp_idx)
    
    with st.expander("⚙️ 数据清洗 (高级)", expanded=False):
        st.markdown("**缺失值 & 归一化**")
        miss_th = st.slider("剔除缺失率 > X", 0.0, 1.0, 0.5, 0.1)
        impute_m = st.selectbox("填充方法", ["min", "mean", "zero"], index=0)
        norm_m = st.selectbox("样本归一化", ["None", "Sum", "Median"], index=0)
        st.markdown("**转化 & 缩放**")
        do_log = st.checkbox("Log2 转化", value=True)
        scale_m = st.selectbox("特征缩放", ["None", "Auto", "Pareto"], index=0)

    all_groups = sorted(raw_df[group_col].astype(str).unique())
    st.divider()
    st.markdown("### 3. 组别筛选")
    selected_groups = st.multiselect("纳入分析的组 (全局):", all_groups, default=all_groups[:2] if len(all_groups)>=2 else all_groups)
    if len(selected_groups) < 2: st.error("至少选 2 个组"); st.stop()
    
    st.markdown("### 4. 差异对比")
    c1, c2 = st.columns(2)
    valid_groups = [g for g in selected_groups]
    case_grp = c1.selectbox("Exp (Case)", valid_groups, index=0)
    ctrl_grp = c2.selectbox("Ctrl (Ref)", valid_groups, index=1 if len(valid_groups)>1 else 0)
    
    st.divider()
    st.subheader("5. 绘图参数")
    p_th = st.number_input("P-value 阈值", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC 阈值", 1.0)
    enable_jitter = st.checkbox("火山图抖动 (Jitter)", value=True)

# ==========================================
# 3. 数据处理
# ==========================================
df_proc, feats = data_cleaning_pipeline(
    raw_df, group_col, missing_thresh=miss_th, impute_method=impute_m, 
    norm_method=norm_m, log_transform=do_log, scale_method=scale_m
)
df_sub = df_proc[df_proc[group_col].isin(selected_groups)].copy()

if case_grp != ctrl_grp:
    res_stats = run_pairwise_statistics(df_sub, group_col, case_grp, ctrl_grp, feats)
    res_stats['Sig'] = 'NS'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
    sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()
else:
    res_stats = pd.DataFrame(); sig_metabolites = []

# ==========================================
# 4. 结果展示
# ==========================================
st.title("📊 代谢组学分析报告")
st.caption(f"对比: {case_grp} vs {ctrl_grp} | 显著差异物: {len(sig_metabolites)} 个")

tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP 特征", "🌋 火山图", "🔥 热图", "📑 详情"])

# --- Tab 1: PCA ---
with tabs[0]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X = StandardScaler().fit_transform(df_sub[feats])
            pca = PCA(n_components=2).fit(X)
            pcs = pca.transform(X)
            var = pca.explained_variance_ratio_
            fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col],
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
            st.plotly_chart(fig_pca, use_container_width=False)

# --- Tab 2: PLS-DA ---
with tabs[1]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if len(df_sub) < 3: st.warning("样本不足")
        else:
            X_pls = StandardScaler().fit_transform(df_sub[feats])
            y_labels = pd.factorize(df_sub[group_col])[0]
            pls_model = PLSRegression(n_components=2).fit(X_pls, y_labels)
            pls_scores = pls_model.x_scores_
            plot_df = pd.DataFrame({'C1': pls_scores[:,0], 'C2': pls_scores[:,1], 'Group': df_sub[group_col].values})
            fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            for i, grp in enumerate(selected_groups):
                sub_g = plot_df[plot_df['Group'] == grp]
                if len(sub_g) >= 3:
                    ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
                    if ell_x is not None:
                        color = GROUP_COLORS[i % len(GROUP_COLORS)]
                        fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', line=dict(color=color, width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
            fig_pls.update_traces(marker=dict(size=14, line=dict(width=1.5, color='black'), opacity=1.0))
            update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
            st.plotly_chart(fig_pls, use_container_width=False)

# --- Tab 3: VIP ---
with tabs[2]:
    st.markdown("### Top 25 VIP Features")
    if 'pls_model' in locals():
        vip_scores = calculate_vips(pls_model)
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
        top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            fig_vip = px.bar(top_vip, x="VIP", y="Metabolite", orientation='h',
                             color="VIP", color_continuous_scale="RdBu_r", width=800, height=700)
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_vip.update_traces(marker_line_color='black', marker_line_width=1.0)
            fig_vip.update_layout(
                template="simple_white", width=800, height=700,
                title={'text': "VIP Scores (PLS-DA)", 'x':0.5, 'xanchor': 'center', 'font': dict(size=20, family="Arial, bold")},
                xaxis=dict(title="VIP Score", showline=True, mirror=True, linewidth=2, linecolor='black'),
                yaxis=dict(title="", showline=True, mirror=True, linewidth=2, linecolor='black'),
                coloraxis_showscale=False,
                margin=dict(l=200, r=40, t=60, b=60) 
            )
            st.plotly_chart(fig_vip, use_container_width=False)

# --- Tab 4: 火山图 ---
with tabs[3]:
    if case_grp == ctrl_grp: st.warning("请选择不同的组")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            plot_df = res_stats.copy()
            x_c, y_c = "Log2_FC", "-Log10_P"
            if enable_jitter:
                np.random.seed(42)
                xr, yr = (plot_df[x_c].max()-plot_df[x_c].min()) or 1, (plot_df[y_c].max()-plot_df[y_c].min()) or 1
                plot_df['Log2_FC_J'] = plot_df[x_c] + np.random.normal(0, xr*0.015, len(plot_df))
                plot_df['-Log10_P_J'] = plot_df[y_c] + np.random.normal(0, yr*0.015, len(plot_df))
                x_c, y_c = "Log2_FC_J", "-Log10_P_J"
            
            fig_vol = px.scatter(plot_df, x=x_c, y=y_c, color="Sig", color_discrete_map=COLOR_PALETTE,
                                 hover_data={"Metabolite":True, "Log2_FC":':.2f', "P_Value":':.2e', x_c:False, y_c:False},
                                 width=600, height=600)
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.update_traces(marker=dict(size=10, opacity=0.75, line=dict(width=1, color='black')))
            update_layout_square(fig_vol, f"Volcano: {case_grp} vs {ctrl_grp}", "Log2 Fold Change", "-Log10(P-value)")
            st.plotly_chart(fig_vol, use_container_width=False)

# --- Tab 5: 热图 ---
with tabs[4]:
    if not sig_metabolites: st.info("无显著差异物")
    else:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            top_n = 50
            top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
            hm_data = df_sub.set_index(group_col)[top_feats]
            lut = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(df_sub[group_col].unique())}
            row_colors = df_sub[group_col].map(lut)
            try:
                g = sns.clustermap(hm_data.astype(float), z_score=1, cmap="vlag", center=0, 
                                   row_colors=row_colors, figsize=(12, 12), 
                                   dendrogram_ratio=(.15, .15), 
                                   cbar_pos=(0.3, 1.02, 0.4, 0.03), cbar_kws={'orientation': 'horizontal'})
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", fontsize=10)
                g.ax_heatmap.set_yticklabels([]); g.ax_heatmap.set_ylabel("Samples", fontsize=12)
                st.pyplot(g.fig)
            except Exception as e: st.error(f"绘图错误: {e}")

# --- Tab 6: 详情 & 箱线图 ---
with tabs[5]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("统计表")
        if not res_stats.empty:
            display_df = res_stats.sort_values("P_Value").copy()
            st.dataframe(display_df.style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"})
                         .background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
                         use_container_width=True, height=600)
    with c2:
        st.subheader("箱线图")
        feat_options = sorted(feats)
        def_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0
        target_feat = st.selectbox("选择代谢物", feat_options, index=def_ix)
        if target_feat:
            box_df = df_sub[[group_col, target_feat]].copy()
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             color_discrete_sequence=GROUP_COLORS, points="all", width=500, height=500)
            fig_box.update_traces(width=0.6, marker=dict(size=7, opacity=0.6, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
            update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
            st.plotly_chart(fig_box, use_container_width=False)

