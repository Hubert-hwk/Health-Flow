import React, { useState } from 'react';
import { uploadReport } from '../api.js';

// 报告类型选项（与后端约定一致）
const REPORT_TYPES = ['体检', '门诊', '住院', '其他'];

// 异常标记展示：H=偏高 L=偏低 N=正常，其余原样显示
function abnormalBadge(flag) {
  if (!flag) return <span className="badge badge-normal">N</span>;
  const f = String(flag).toUpperCase();
  if (f === 'H' || f === 'HIGH' || f === '高') return <span className="badge badge-high">H</span>;
  if (f === 'L' || f === 'LOW' || f === '低') return <span className="badge badge-low">L</span>;
  if (f === 'N' || f === 'NORMAL' || f === '正常') return <span className="badge badge-normal">N</span>;
  return <span className="badge">{String(flag)}</span>;
}

export default function Upload() {
  const [patientId, setPatientId] = useState('');
  const [reportType, setReportType] = useState('体检');
  const [department, setDepartment] = useState('');
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // 选择文件时做基础校验
  const onFileChange = (e) => {
    const f = e.target.files && e.target.files[0];
    setFile(f || null);
    setError('');
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    // 必填校验
    if (!patientId.trim()) {
      setError('请填写患者编号（patient_id）');
      return;
    }
    if (!file) {
      setError('请选择要上传的报告文件');
      return;
    }

    const formData = new FormData();
    formData.append('patient_id', patientId.trim());
    formData.append('file', file);
    formData.append('report_type', reportType);
    if (department.trim()) formData.append('department', department.trim());

    setSubmitting(true);
    try {
      const data = await uploadReport(formData);
      setResult(data);
    } catch (err) {
      // 413/415/422 等错误信息由后端 detail 字段给出，直接展示
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const metrics = (result && result.metrics) || [];

  return (
    <div>
      <div className="card">
        <h3 className="card-title">上传体检/门诊/住院报告</h3>
        <form onSubmit={onSubmit} className="form">
          <div className="field">
            <label htmlFor="patient_id">患者编号（必填）</label>
            <input
              id="patient_id"
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="例如 P001"
            />
          </div>

          <div className="field">
            <label htmlFor="report_type">报告类型</label>
            <select id="report_type" value={reportType} onChange={(e) => setReportType(e.target.value)}>
              {REPORT_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="department">科室（可选）</label>
            <input
              id="department"
              type="text"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              placeholder="例如 心血管内科"
            />
          </div>

          <div className="field">
            <label htmlFor="file">报告文件（必选）</label>
            <input
              id="file"
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.gif,.bmp"
              onChange={onFileChange}
            />
            <p className="hint">支持格式：PDF / JPG / JPEG / PNG / GIF / BMP</p>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? '上传解析中…' : '上传并解析'}
          </button>
        </form>
      </div>

      {/* 上传成功后的解析结果 */}
      {result && (
        <div className="card">
          <h3 className="card-title">解析结果</h3>
          <div className="meta-line">
            <span>报告 ID：<strong>{result.id}</strong></span>
            <span>类型：{result.report_type || '—'}</span>
            <span>检查日期：{result.exam_date || '—'}</span>
            <span>科室：{result.department || '—'}</span>
          </div>
          {metrics.length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>指标名称</th>
                  <th>数值</th>
                  <th>单位</th>
                  <th>参考范围</th>
                  <th>异常</th>
                  <th>页码</th>
                  <th>证据文本</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td>{m.metric_name || '—'}</td>
                    <td>{m.metric_value ?? '—'}</td>
                    <td>{m.unit || '—'}</td>
                    <td>{m.reference_range || '—'}</td>
                    <td>{abnormalBadge(m.abnormal_flag)}</td>
                    <td>{m.page_number ?? '—'}</td>
                    <td className="evidence-cell">{m.evidence_text || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">该报告未解析出指标数据。</p>
          )}
        </div>
      )}
    </div>
  );
}
