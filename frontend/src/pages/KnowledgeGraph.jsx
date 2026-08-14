import React, { useState } from 'react';
import { kgDepartment } from '../api.js';

// 知识图谱页：输入症状，查询其对应的分诊科室
export default function KnowledgeGraph() {
  const [symptom, setSymptom] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const search = async (e) => {
    if (e) e.preventDefault();
    const s = symptom.trim();
    if (!s) {
      setError('请输入症状关键词');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await kgDepartment(s);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // 判断图谱是否返回了有效数据
  const isEmpty = result && !result.department && !result.source && Object.keys(result).length <= 2;

  return (
    <div>
      <div className="card">
        <h3 className="card-title">症状 → 科室 知识图谱查询</h3>
        <form onSubmit={search} className="form-inline">
          <input
            type="text"
            value={symptom}
            onChange={(e) => setSymptom(e.target.value)}
            placeholder="输入症状，例如 胸痛、头晕、发热"
          />
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? '查询中…' : '查询'}
          </button>
        </form>
        {error && <div className="alert alert-error">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <h3 className="card-title">查询结果</h3>

          {/* 图谱节点可视化：症状 → 科室 */}
          <div className="kg-flow">
            <div className="kg-node kg-node-source">
              <div className="kg-node-label">症状</div>
              <div className="kg-node-value">{result.symptom || symptom}</div>
            </div>
            <div className="kg-arrow">→</div>
            <div className="kg-node kg-node-target">
              <div className="kg-node-label">分诊科室</div>
              <div className="kg-node-value">{result.department || '未命中'}</div>
            </div>
            {result.source && (
              <>
                <div className="kg-arrow">←</div>
                <div className="kg-node kg-node-source">
                  <div className="kg-node-label">来源</div>
                  <div className="kg-node-value">{result.source}</div>
                </div>
              </>
            )}
          </div>

          {isEmpty && (
            <p className="hint">
              注：后端未返回有效的科室映射（图谱数据可能为空）。可尝试其他症状关键词，或检查知识图谱数据是否已导入。
            </p>
          )}

          <div className="meta-section">
            <div className="meta-label">原始响应</div>
            <pre className="json-block">{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
