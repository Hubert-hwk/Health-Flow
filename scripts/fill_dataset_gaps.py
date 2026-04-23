"""使用 MiniMax LLM 补全 SFT 数据集缺口。

用法:
    python scripts/fill_dataset_gaps.py --api-key YOUR_KEY --target examination_report --count 500
    python scripts/fill_dataset_gaps.py --api-key YOUR_KEY --target all
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from app.service.llm_expander import LLMExpander, LLMExpanderConfig
from app.config import get_settings

# ===== 模板定义 =====

EXAMINATION_TEMPLATES = [
    {
        "template": "请解读这份体检报告中{metric}指标的意义。{metric}: {value} {unit}, 参考范围: {range}",
        "placeholders": {
            "metric": ["空腹血糖", "餐后2小时血糖", "糖化血红蛋白", "血压", "总胆固醇", "甘油三酯", "低密度脂蛋白", "高密度脂蛋白", "尿酸", "肌酐", "谷丙转氨酶", "谷草转氨酶", "总胆红素", "白细胞", "血红蛋白", "血小板", "红细胞"],
            "value": ["6.5", "8.2", "7.1", "6.8", "142/88", "138/85", "155/95", "160/100", "5.8", "6.2", "2.1", "3.5", "3.8", "4.2", "1.85", "1.92", "220", "480", "52", "38", "95", "68", "15.8", "12.5", "4.5", "3.8", "185", "105", "3.5", "138", "8.2", "7.5"],
            "unit": ["mmol/L", "mmol/L", "%", "mmHg", "mmHg", "mmol/L", "mmol/L", "mmol/L", "μmol/L", "μmol/L", "U/L", "U/L", "μmol/L", "×10^9/L", "g/L", "×10^12/L", "×10^9/L"],
            "range": ["3.9-6.1", "<7.8", "4.0-6.0", "<140/90", "120-139/80-89", "<5.18", "<1.7", "<3.4", "0.8-1.8", "1.0-1.7", "208-428", "44-133", "9-50", "15-40", "3.5-9.5", "120-160", "4.3-5.8", "125-350", "130-175"]
        }
    },
    {
        "template": "我的体检报告显示{metric}偏高，是什么问题？{metric}: {value} {unit}",
        "placeholders": {
            "metric": ["空腹血糖", "血压", "尿酸", "总胆固醇", "甘油三酯", "谷丙转氨酶", "谷草转氨酶"],
            "value": ["6.5", "7.2", "8.5", "145/95", "155/98", "520", "560", "5.8", "6.2", "5.6", "2.1", "3.5", "58", "65", "48"],
            "unit": ["mmol/L", "mmol/L", "mmHg", "μmol/L", "mmol/L", "mmol/L", "U/L", "U/L"],
            "range": ["3.9-6.1", "<140/90", "208-428", "<5.18", "<1.7", "9-50", "15-40"]
        }
    },
    {
        "template": "体检发现{metric}异常，需要进一步检查吗？指标：{metric} {value} {unit}，参考值：{range}",
        "placeholders": {
            "metric": ["CEA", "AFP", "CA199", "CA125", "CA724", "NSE", "CYFRA21-1", "SCC"],
            "value": ["6.5", "8.2", "12.5", "38", "39", "18", "3.5", "2.1", "15.8", "25"],
            "unit": ["ng/mL", "U/mL", "μg/L"],
            "range": ["<5.0", "<37", "<35", "<35", "<6.9", "<15.2", "<3.3", "<1.5"]
        }
    }
]

METRIC_QUERY_TEMPLATES = [
    {
        "template": "{metric}偏高饮食需要注意什么？{metric}: {value} {unit}（参考范围 {range}）",
        "placeholders": {
            "metric": ["空腹血糖", "尿酸", "总胆固醇", "甘油三酯", "低密度脂蛋白", "血压"],
            "value": ["6.5", "7.2", "6.8", "520", "480", "5.6", "5.8", "4.2", "3.8", "145/95", "138/88"],
            "unit": ["mmol/L", "μmol/L", "mmol/L", "mmHg"],
            "range": ["3.9-6.1", "208-428", "<5.18", "<1.7", "<3.4", "<140/90"]
        }
    },
    {
        "template": "体检指标{method}偏高吃什么好？{metric}: {value} {unit}",
        "placeholders": {
            "metric": ["血红蛋白", "白细胞", "血小板", "红细胞"],
            "method": ["轻微", "明显", "轻度"],
            "value": ["95", "105", "110", "3.8", "3.5", "4.0", "180", "200", "220", "4.2", "4.5", "5.0"],
            "unit": ["g/L", "×10^9/L", "×10^12/L", "×10^9/L"]
        }
    },
    {
        "template": "{metric}高的人不能吃什么？{metric}: {value} {unit}",
        "placeholders": {
            "metric": ["尿酸", "胆固醇", "甘油三酯", "血糖"],
            "value": ["480", "520", "560", "5.6", "5.8", "6.2", "5.3", "4.2", "3.8"],
            "unit": ["μmol/L", "mmol/L"]
        }
    }
]

TRIAGE_TEMPLATES = [
    {
        "template": "最近出现{symptom}，应该挂什么科？症状：{symptom_detail}",
        "symptoms": [
            ("胸闷心悸", "胸闷、心悸、活动后气促，持续1周"),
            ("头晕头痛", "经常头晕，偶有头痛，血压偏高"),
            ("关节疼痛", "右膝关节肿胀、疼痛，活动受限3天"),
            ("腹痛腹泻", "上腹部隐痛伴腹泻2周，大便不成形"),
            ("咳嗽咳痰", "咳嗽伴黄痰、低热3天"),
            ("失眠焦虑", "失眠多梦、情绪焦虑，入睡困难"),
            ("皮疹瘙痒", "四肢出现红色皮疹，伴瘙痒"),
            ("体检肺结节", "体检发现肺部有阴影，直径5mm"),
            ("尿频尿急", "尿频尿急2天，伴小腹不适"),
            ("体重下降", "半年内体重下降5公斤，无明显原因"),
            ("淋巴结肿大", "颈部淋巴结肿大2周，无疼痛"),
            ("乳腺肿块", "乳房发现肿块，边界不清"),
            ("牙龈出血", "牙龈经常出血，刷牙时明显"),
            ("耳鸣听力下降", "右耳耳鸣伴听力下降"),
            ("视物模糊", "视力逐渐模糊，眼前有漂浮物"),
            ("手足麻木", "手脚麻木感，走路有踩棉花感"),
            ("骨质疏松", "腰背疼痛，体检查出骨密度降低"),
            ("甲状腺结节", "体检发现甲状腺结节，边界清晰"),
        ]
    },
    {
        "template": "我应该去哪个科室看病？症状：{symptom}",
        "symptoms": [
            ("胃酸胃胀", "胃酸过多，腹胀，饭后明显"),
            ("脂肪肝", "B超显示中度脂肪肝，肝功能异常"),
            ("肾结石", "腰部绞痛，伴血尿"),
            ("带状疱疹", "腰部出现成簇水泡，疼痛明显"),
            ("腰椎间盘突出", "腰痛伴左下肢放射痛，咳嗽时加重"),
            ("痛风发作", "右足拇指关节红肿热痛"),
            ("焦虑抑郁", "情绪低落，对什么都不感兴趣"),
            ("湿疹皮炎", "皮肤瘙痒起疹，反复发作"),
            ("卵巢囊肿", "B超发现卵巢囊肿，3cm大小"),
            ("前列腺增生", "尿频尿急夜尿多，排尿困难"),
        ]
    }
]


def load_existing_counts(category: str) -> int:
    """从现有文件加载各类别已有数量。"""
    training_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sft', 'training_data.jsonl')
    safety_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sft', 'safety_qa.jsonl')

    counts = Counter()
    if os.path.exists(training_file):
        with open(training_file, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    counts[r.get('category', '')] += 1
                except:
                    pass
    if os.path.exists(safety_file):
        with open(safety_file, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('category') == '医疗安全问答':
                        counts['医疗安全问答'] += 1
                except:
                    pass

    return counts.get(category, 0)


def generate_for_category(category: str, target_count: int, api_key: str, batch_size: int = 5, max_items: int = None):
    """为指定类别生成数据。"""
    settings = get_settings()
    settings.MINIMAX_API_KEY = api_key

    config = LLMExpanderConfig(
        model="MiniMax-M2.7",
        temperature=0.7,
        max_tokens=1024,
        batch_size=batch_size,
        max_retries=5
    )
    expander = LLMExpander(config=config)

    if category == "examination_report" or category == "体检报告解读":
        templates = EXAMINATION_TEMPLATES
        per_template = (target_count + len(templates) - 1) // len(templates)
        results = []
        for i, tmpl in enumerate(templates):
            logger.info(f"[{i+1}/{len(templates)}] 生成体检报告解读模板 {i+1}")
            batch_count = min(per_template, max_items or per_template)
            batch_results = expander.expand_examination(
                template=tmpl["template"],
                placeholders=tmpl["placeholders"],
                count=batch_count
            )
            results.extend(batch_results)
            logger.info(f"    获得 {len(batch_results)} 条")
            time.sleep(1)  # 避免过快
        return results

    elif category == "metric_query" or category == "指标异常问询":
        templates = METRIC_QUERY_TEMPLATES
        per_template = (target_count + len(templates) - 1) // len(templates)
        results = []
        for i, tmpl in enumerate(templates):
            logger.info(f"[{i+1}/{len(templates)}] 生成指标异常问询模板 {i+1}")
            batch_count = min(per_template, max_items or per_template)
            batch_results = expander.expand_metric_query(
                template=tmpl["template"],
                placeholders=tmpl["placeholders"],
                count=batch_count
            )
            results.extend(batch_results)
            logger.info(f"    获得 {len(batch_results)} 条")
            time.sleep(1)
        return results

    elif category == "triage" or category == "科室分诊建议":
        templates = TRIAGE_TEMPLATES
        results = []
        for i, tmpl in enumerate(templates):
            logger.info(f"[{i+1}/{len(templates)}] 生成科室分诊模板 {i+1}")
            batch_results = expander.expand_triage(
                template=tmpl["template"],
                symptoms=tmpl.get("symptoms", []),
                count=min(target_count, len(tmpl.get("symptoms", [])) * 10)
            )
            results.extend(batch_results)
            logger.info(f"    获得 {len(batch_results)} 条")
            time.sleep(1)
        return results

    return []


def save_results(results: list, category: str, output_file: str):
    """保存结果到JSONL文件。"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for r in results:
            record = r.to_dict() if hasattr(r, 'to_dict') else r
            # 确保免责提示
            if 'output' in record and 'output' in record:
                out = record['output']
                if not any(d in out for d in ['仅供参考', '请咨询专业医生', '遵医嘱']):
                    record['output'] = out + ' 仅供参考，请咨询专业医生。'
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='补全SFT数据集缺口')
    parser.add_argument('--api-key', required=True, help='MiniMax API Key')
    parser.add_argument('--target', choices=['examination_report', 'metric_query', 'triage', 'all'], default='all')
    parser.add_argument('--batch-size', type=int, default=5, help='每批生成数量')
    parser.add_argument('--count', type=int, default=None, help='指定生成数量')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sft')
    training_file = os.path.join(data_dir, 'training_data.jsonl')
    output_file = os.path.join(data_dir, 'training_data.jsonl')

    # 计算各类别目标缺口
    targets = {
        '体检报告解读': 3000,
        '指标异常问询': 2500,
        '科室分诊建议': 1500,
    }

    existing = Counter()
    if os.path.exists(training_file):
        with open(training_file, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing[r.get('category', '')] += 1
                except:
                    pass

    categories_to_generate = []
    if args.target == 'all':
        categories_to_generate = [
            ('examination_report', '体检报告解读'),
            ('metric_query', '指标异常问询'),
            ('triage', '科室分诊建议'),
        ]
    elif args.target == 'examination_report':
        categories_to_generate = [('examination_report', '体检报告解读')]
    elif args.target == 'metric_query':
        categories_to_generate = [('metric_query', '指标异常问询')]
    elif args.target == 'triage':
        categories_to_generate = [('triage', '科室分诊建议')]

    for cat_key, cat_name in categories_to_generate:
        current = existing.get(cat_name, 0)
        target = targets[cat_name]
        gap = max(0, target - current)
        if gap == 0:
            logger.info(f'【{cat_name}】已满足目标 ({current}/{target})，跳过')
            continue

        count = args.count if args.count else gap
        logger.info(f'【{cat_name}】当前{current}条，目标{target}条，生成{count}条...')

        results = generate_for_category(cat_key, count, args.api_key, batch_size=args.batch_size)
        logger.info(f'共获得 {len(results)} 条结果')

        if results:
            # 追加到training_data.jsonl
            mode = 'a' if os.path.exists(output_file) else 'w'
            with open(output_file, mode, encoding='utf-8') as f:
                for r in results:
                    record = r.to_dict() if hasattr(r, 'to_dict') else r
                    record['category'] = cat_name
                    # 确保免责提示
                    out = record.get('output', '')
                    if out and not any(d in out for d in ['仅供参考', '请咨询专业医生', '遵医嘱']):
                        record['output'] = out + ' 仅供参考，请咨询专业医生。'
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f'已保存到 {output_file}')

        logger.info(f'等待2秒后继续...')
        time.sleep(2)

    logger.info('全部完成!')


if __name__ == '__main__':
    main()
