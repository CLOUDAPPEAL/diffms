import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols

# -------------------------- 1. 配置参数 --------------------------
root_dir = r"C:\Users\19691\Desktop\fsdownload\20251208_antibio_test"
output_excel = "result.xlsx"
similarity_methods = {
    "RDK_Tanimoto": lambda fp1, fp2: DataStructs.TanimotoSimilarity(fp1, fp2),
    "RDK_Dice": lambda fp1, fp2: DataStructs.DiceSimilarity(fp1, fp2),
    "Morgan2_Tanimoto": lambda fp1, fp2: DataStructs.TanimotoSimilarity(fp1, fp2),
    "Morgan2_Cosine": lambda fp1, fp2: DataStructs.CosineSimilarity(fp1, fp2),
    "MACCS_Tanimoto": lambda fp1, fp2: DataStructs.TanimotoSimilarity(fp1, fp2)
}

# -------------------------- 2. 核心工具函数 --------------------------
def load_pkl_file(file_path):
    """加载pkl文件，返回Mol对象列表"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载文件失败 {file_path}：{str(e)}")
        return None

def fix_mol_ring_info(mol):
    """修复Mol对象的环信息初始化问题"""
    if mol is None:
        return None
    # 强制计算环信息
    try:
        Chem.GetSSSR(mol)  # 计算最小环基集，初始化RingInfo
        mol.UpdatePropertyCache()  # 更新属性缓存
        return mol
    except Exception as e:
        print(f"修复Mol环信息失败：{str(e)}")
        return None

def validate_mol(mol):
    """校验并修复Mol对象的有效性"""
    if mol is None:
        return None
    # 检查SMILES是否有效
    try:
        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return None
    except:
        return None
    # 修复环信息
    mol = fix_mol_ring_info(mol)
    return mol

def get_mol_fingerprints(mol):
    """生成分子的多种指纹（修复环信息问题）"""
    # 先校验并修复Mol对象
    mol = validate_mol(mol)
    if mol is None:
        return None
    
    try:
        # 1. RDK指纹（基础指纹）
        rdk_fp = Chem.RDKFingerprint(mol)
        # 2. Morgan指纹（半径2，1024位）- 修复环信息后可正常计算
        morgan2_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # 3. MACCS指纹（166位）
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        
        return {
            "RDK": rdk_fp,
            "Morgan2": morgan2_fp,
            "MACCS": maccs_fp
        }
    except Exception as e:
        print(f"生成指纹失败：{str(e)}")
        return None

def calculate_all_similarities(true_mol, pred_mol):
    """计算true和pred分子的所有相似度类型，返回字典"""
    sim_results = {}
    # 获取指纹（自动修复环信息）
    true_fps = get_mol_fingerprints(true_mol)
    pred_fps = get_mol_fingerprints(pred_mol)
    
    if not true_fps or not pred_fps:
        # 无效分子，所有相似度返回None
        for method in similarity_methods.keys():
            sim_results[method] = None
        return sim_results
    
    # 逐个计算相似度
    for method_name, calc_fun in similarity_methods.items():
        fp_type = method_name.split("_")[0]  # 提取指纹类型（RDK/Morgan2/MACCS）
        try:
            sim = calc_fun(true_fps[fp_type], pred_fps[fp_type])
            sim_results[method_name] = round(sim, 4)  # 保留4位小数
        except Exception as e:
            print(f"计算{method_name}相似度失败：{str(e)}")
            sim_results[method_name] = None
    return sim_results

def get_mol_info(mol):
    """提取分子基础信息（SMILES、化学式、分子量）"""
    # 先校验并修复Mol对象
    mol = validate_mol(mol)
    if mol is None:
        return {"SMILES": None, "化学式": None, "分子量": None}
    
    try:
        smiles = Chem.MolToSmiles(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
        return {
            "SMILES": smiles,
            "化学式": formula,
            "分子量": mw
        }
    except Exception as e:
        print(f"提取分子信息失败：{str(e)}")
        return {"SMILES": None, "化学式": None, "分子量": None}

# -------------------------- 3. 批量处理文件 --------------------------
def process_all_files():
    """处理所有true/pred配对文件，返回结果列表"""
    results = []
    file_index = 0
    
    while True:
        # 构建文件路径
        true_file = os.path.join(root_dir, f"dev_rank_0_true_{file_index}.pkl")
        pred_file = os.path.join(root_dir, f"dev_rank_0_pred_{file_index}.pkl")
        
        # 终止条件：true或pred文件不存在
        if not os.path.exists(true_file) or not os.path.exists(pred_file):
            if file_index == 0:
                print("未找到任何true/pred配对文件！")
            else:
                print(f"\n已处理完所有文件（最后索引：{file_index-1}）")
            break
        
        print(f"\n开始处理文件组：{file_index}")
        print(f"True文件：{true_file}")
        print(f"Pred文件：{pred_file}")
        
        # 加载文件
        true_mols = load_pkl_file(true_file)
        pred_mols = load_pkl_file(pred_file)
        
        if true_mols is None or pred_mols is None:
            file_index += 1
            continue
        
        # 验证数据格式
        if not isinstance(true_mols, list):
            print(f"True文件{file_index}格式错误：非列表类型")
            file_index += 1
            continue
        if not isinstance(pred_mols, list):
            print(f"Pred文件{file_index}格式错误：非列表类型")
            file_index += 1
            continue
        
        # 遍历每个true分子（对应10个pred分子）
        for true_idx, true_mol in enumerate(true_mols):
            # 检查pred分子数量是否匹配（每个true对应10个pred）
            if true_idx >= len(pred_mols):
                print(f"警告：文件组{file_index}中，true分子{true_idx}无对应pred分子")
                continue
            pred_mol_group = pred_mols[true_idx]
            
            if not isinstance(pred_mol_group, list) or len(pred_mol_group) != 10:
                print(f"警告：文件组{file_index}中，true分子{true_idx}的pred分子数量≠10，跳过")
                continue
            
            # 遍历该true分子对应的10个pred分子
            for pred_sub_idx, pred_mol in enumerate(pred_mol_group):
                # 提取分子信息（自动修复Mol对象）
                true_info = get_mol_info(true_mol)
                pred_info = get_mol_info(pred_mol)
                
                # 计算所有相似度（自动修复Mol对象）
                sim_results = calculate_all_similarities(true_mol, pred_mol)
                
                # 组装结果行
                result_row = {
                    "文件组索引": file_index,
                    "True分子索引": true_idx,
                    "Pred分子子索引": pred_sub_idx,
                    # True分子信息
                    "True_SMILES": true_info["SMILES"],
                    "True_化学式": true_info["化学式"],
                    "True_分子量": true_info["分子量"],
                    # Pred分子信息
                    "Pred_SMILES": pred_info["SMILES"],
                    "Pred_化学式": pred_info["化学式"],
                    "Pred_分子量": pred_info["分子量"]
                }
                
                # 添加相似度结果
                result_row.update(sim_results)
                results.append(result_row)
                
                # 打印进度
                if (len(results) % 10) == 0:
                    print(f"已处理 {len(results)} 组分子对")
        
        file_index += 1
    
    return results

# -------------------------- 4. 主执行流程 --------------------------
if __name__ == "__main__":
    # 处理所有文件
    all_results = process_all_files()
    
    if not all_results:
        print("无有效结果，退出程序")
    else:
        # 转换为DataFrame并保存Excel
        df = pd.DataFrame(all_results)
        df.to_excel(os.path.join(root_dir, output_excel), index=False, engine="openpyxl")
        print(f"\n✅ 结果已保存至：{os.path.join(root_dir, output_excel)}")
        print(f"📊 共处理 {len(all_results)} 组true-pred分子对")
        
        # 输出相似度统计信息
        sim_cols = list(similarity_methods.keys())
        valid_sim_counts = {col: df[col].notna().sum() for col in sim_cols}
        print("\n📈 各相似度有效计算数量：")
        for col, count in valid_sim_counts.items():
            print(f"  {col}: {count} / {len(all_results)}")