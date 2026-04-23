"""使用 MiniMax LLM 补全体检报告解读和指标异常问询数据集。"""

import os, sys, json, time, re
from collections import Counter

sys.path.insert(0, '.')
os.environ['MINIMAX_API_KEY'] = 'sk-cp-0MBJz7YZPEgIZgCbFZOZ2ccAbL5QH72sMqcuk_WLwsyFSnt_-FtVCDLdMXILxTv0cmmE25aSGpTHO2gZweV5ZQcv97EcQWkL_3YG8RWI3YlnXPMxSs0aO8U'

from app.service.llm_expander import LLMExpander, LLMExpanderConfig

OUTPUT_FILE = 'data/sft/training_data.jsonl'
os.makedirs('data/sft', exist_ok=True)

def ensure_disclaimer(output):
    if output and not any(d in output for d in ['仅供参考', '请咨询专业医生', '遵医嘱']):
        return output + ' 仅供参考，请咨询专业医生。'
    return output

def load_counts():
    counts = Counter()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    counts[r.get('category', '')] += 1
                except:
                    pass
    return counts

def generate_examination(target_count, batch_size=5):
    """生成体检报告解读数据。"""
    templates = [
        {
            'template': '请解读这份体检报告中{metric}指标的意义。{metric}: {value} {unit}, 参考范围: {range}',
            'placeholders': {
                'metric': ['空腹血糖', '餐后2小时血糖', '糖化血红蛋白', '血压', '总胆固醇', '甘油三酯', '低密度脂蛋白', '高密度脂蛋白', '尿酸', '肌酐', '谷丙转氨酶', '谷草转氨酶', '总胆红素', '白细胞', '血红蛋白', '血小板'],
                'value': ['6.5', '7.2', '8.5', '142/88', '138/85', '155/95', '5.8', '6.2', '2.1', '3.5', '5.6', '2.1', '1.85', '520', '480', '52', '38', '95', '68', '15.8', '12.5', '185', '105'],
                'unit': ['mmol/L', 'mmHg', '%', 'μmol/L', 'U/L', 'g/L', '×10^9/L', '×10^12/L'],
                'range': ['3.9-6.1', '<7.8', '4.0-6.0', '<140/90', '120-139/80-89', '<5.18', '<1.7', '<3.4', '0.8-1.8', '208-428', '44-133', '9-50', '15-40', '3.5-9.5', '120-160', '125-350']
            }
        },
        {
            'template': '我的体检报告显示{metric}偏高，是什么问题？{metric}: {value} {unit}，参考值：{range}',
            'placeholders': {
                'metric': ['空腹血糖', '血压', '尿酸', '总胆固醇', '甘油三酯', '谷丙转氨酶', '谷草转氨酶'],
                'value': ['6.5', '7.2', '8.5', '145/95', '155/98', '520', '480', '5.6', '5.8', '58', '65', '48'],
                'unit': ['mmol/L', 'mmHg', 'μmol/L', 'U/L'],
                'range': ['3.9-6.1', '<140/90', '208-428', '<5.18', '<1.7', '9-50', '15-40']
            }
        },
        {
            'template': '体检发现{metric}异常，需要进一步检查吗？指标：{metric} {value} {unit}，参考值：{range}',
            'placeholders': {
                'metric': ['CEA', 'AFP', 'CA199', 'CA125', 'CA724', 'NSE', 'CYFRA21-1', 'SCC'],
                'value': ['6.5', '8.2', '12.5', '38', '39', '18', '3.5', '2.1', '15.8'],
                'unit': ['ng/mL', 'U/mL', 'μg/L'],
                'range': ['<5.0', '<37', '<35', '<35', '<6.9', '<15.2', '<3.3', '<1.5']
            }
        }
    ]

    config = LLMExpanderConfig(model='MiniMax-M2.7', batch_size=batch_size, max_tokens=512, max_retries=3, temperature=0.8)
    expander = LLMExpander(config=config)

    generated = 0
    per_template = (target_count + len(templates) - 1) // len(templates)

    for ti, tmpl in enumerate(templates):
        if generated >= target_count:
            break
        print(f'  模板{ti+1}/{len(templates)}...')
        batch_count = min(per_template, 100)
        results = expander.expand_examination(tmpl['template'], tmpl['placeholders'], count=batch_count)
        print(f'    获得 {len(results)} 条')

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for r in results:
                rec = r.to_dict()
                rec['category'] = '体检报告解读'
                rec['output'] = ensure_disclaimer(rec.get('output', ''))
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                generated += 1
                if generated >= target_count:
                    break
        time.sleep(1)

    return generated

def generate_metric_query(target_count, batch_size=5):
    """生成指标异常问询数据。"""
    templates = [
        {
            'template': '{metric}偏高饮食需要注意什么？{metric}: {value} {unit}（参考范围 {range}）',
            'placeholders': {
                'metric': ['空腹血糖', '尿酸', '总胆固醇', '甘油三酯', '低密度脂蛋白', '血压'],
                'value': ['6.5', '7.2', '520', '480', '5.6', '5.8', '4.2', '145/95', '138/88'],
                'unit': ['mmol/L', 'μmol/L', 'mmol/L', 'mmHg'],
                'range': ['3.9-6.1', '208-428', '<5.18', '<1.7', '<3.4', '<140/90']
            }
        },
        {
            'template': '体检指标{method}偏高吃什么好？{metric}: {value} {unit}',
            'placeholders': {
                'metric': ['血红蛋白', '白细胞', '血小板', '红细胞'],
                'method': ['轻微', '明显', '轻度'],
                'value': ['95', '105', '110', '3.8', '3.5', '4.0', '180', '200', '220'],
                'unit': ['g/L', '×10^9/L', '×10^12/L']
            }
        },
        {
            'template': '{metric}高的人不能吃什么？{metric}: {value} {unit}',
            'placeholders': {
                'metric': ['尿酸', '胆固醇', '甘油三酯', '血糖'],
                'value': ['480', '520', '560', '5.6', '5.8', '6.2', '5.3', '4.2', '3.8'],
                'unit': ['μmol/L', 'mmol/L']
            }
        }
    ]

    config = LLMExpanderConfig(model='MiniMax-M2.7', batch_size=batch_size, max_tokens=512, max_retries=3, temperature=0.8)
    expander = LLMExpander(config=config)

    generated = 0
    per_template = (target_count + len(templates) - 1) // len(templates)

    for ti, tmpl in enumerate(templates):
        if generated >= target_count:
            break
        print(f'  模板{ti+1}/{len(templates)}...')
        batch_count = min(per_template, 100)
        results = expander.expand_metric_query(tmpl['template'], tmpl['placeholders'], count=batch_count)
        print(f'    获得 {len(results)} 条')

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for r in results:
                rec = r.to_dict()
                rec['category'] = '指标异常问询'
                rec['output'] = ensure_disclaimer(rec.get('output', ''))
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                generated += 1
                if generated >= target_count:
                    break
        time.sleep(1)

    return generated

def main():
    targets = {'体检报告解读': 3000, '指标异常问询': 2500}
    existing = load_counts()

    for cat_name, target in targets.items():
        current = existing.get(cat_name, 0)
        gap = target - current
        if gap <= 0:
            print(f'[{cat_name}] 已满足 ({current}/{target})，跳过')
            continue

        print(f'\n[{cat_name}] 当前{current}条，目标{target}条，生成{gap}条...')

        if cat_name == '体检报告解读':
            # 每批50条，分批生成
            for batch_start in range(0, gap, 50):
                batch_count = min(50, gap - batch_start)
                print(f'  第{batch_start//50+1}批: {batch_count}条')
                n = generate_examination(batch_count, batch_size=5)
                print(f'    实际生成 {n} 条')
                existing[cat_name] += n
                time.sleep(2)
        else:
            for batch_start in range(0, gap, 50):
                batch_count = min(50, gap - batch_start)
                print(f'  第{batch_start//50+1}批: {batch_count}条')
                n = generate_metric_query(batch_count, batch_size=5)
                print(f'    实际生成 {n} 条')
                existing[cat_name] += n
                time.sleep(2)

    print('\n最终统计:')
    for cat, cnt in load_counts().items():
        print(f'  {cat}: {cnt}')

if __name__ == '__main__':
    main()
