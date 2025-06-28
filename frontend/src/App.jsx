// frontend/src/App.jsx
import React, { useState, useEffect, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";
import {
  Upload,
  Key,
  Play,
  FileText,
  Download,
  BarChart2,
  Server,
  X,
  Loader2,
  StopCircle,
  RefreshCw,
  Filter,
  GitCompare,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  Eye,
  Search,
  Moon,
  Sun,
  File,
  CloudUpload,
  FolderOpen,
  Trash2,
  Terminal,
  Activity,
  AlertCircle,
  Info,
} from "lucide-react";
import "./App.css";

// --- Constants for flexibility ---
const STATUS = {
  IDLE: "idle",
  LOADING: "loading",
  SUCCESS: "success",
  ERROR: "error",
  INFO: "info",
  PROCESSING: "processing",
  STOPPED: "stopped",
};

const ALERT_TYPES = {
  INFO: "info",
  ERROR: "error",
  SUCCESS: "success",
  WARNING: "warning",
};

const DEVICE_STATUS = {
  PENDING: "pending",
  CONNECTING: "connecting",
  SUCCESS: "success",
  FAILED: "failed",
  RETRYING: "retrying",
  STOPPED: "stopped",
};

const MESSAGES = {
  INITIALIZING: "Initializing Backend...",
  API_READY: "Ready. Please provide credentials and upload device file.",
  PROCESSING: "Processing, please wait...",
  AWAITING_FILE: "Please upload a CSV file with device list...",
  PROCESS_FINISHED: "Process finished.",
};

// Environment Detection
const isDevelopment =
  process.env.NODE_ENV === "development" ||
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const API_BASE_URL = isDevelopment ? "http://localhost:5000/api" : "/api";

console.log(`Running in ${isDevelopment ? "development" : "production"} mode`);
console.log(`API Base URL: ${API_BASE_URL}`);

// API Helper Functions
const api = {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  },

  async getSystemInfo() {
    if (isDevelopment && window.pywebview) {
      return await window.pywebview.api.get_system_info();
    } else {
      return this.request("/system_info");
    }
  },

  async uploadCsvFile(file) {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const url = isDevelopment
        ? `${API_BASE_URL}/upload_csv`
        : "/api/upload_csv";

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("File upload failed:", error);
      throw error;
    }
  },

  async processDevicesFromFile(
    username,
    password,
    filepath = null,
    retryFailedOnly = false
  ) {
    if (isDevelopment && window.pywebview && !filepath) {
      return await window.pywebview.api.process_devices_from_file(
        username,
        password
      );
    } else {
      return this.request("/process_devices", {
        method: "POST",
        body: JSON.stringify({
          username,
          password,
          filepath,
          retry_failed_only: retryFailedOnly,
        }),
      });
    }
  },

  async getProcessingStatus(sessionId) {
    if (isDevelopment && !sessionId) {
      return null;
    } else {
      return this.request(`/processing_status/${sessionId}`);
    }
  },

  async stopProcessing(sessionId) {
    if (isDevelopment && sessionId) {
      return this.request(`/stop_processing/${sessionId}`, {
        method: "POST",
      });
    } else {
      return null;
    }
  },

  async retryFailedDevices(username, password, results) {
    if (isDevelopment && window.pywebview) {
      // PyWebView implementation would go here
      return null;
    } else {
      return this.request("/retry_failed", {
        method: "POST",
        body: JSON.stringify({ username, password, results }),
      });
    }
  },

  async filterResults(results, filterType, filterValue) {
    if (isDevelopment && window.pywebview) {
      // Local filtering for PyWebView mode
      return { status: "success", data: results };
    } else {
      return this.request("/filter_results", {
        method: "POST",
        body: JSON.stringify({
          results,
          filter_type: filterType,
          filter_value: filterValue,
        }),
      });
    }
  },

  async compareFiles() {
    if (isDevelopment && window.pywebview) {
      return await window.pywebview.api.compare_data_files();
    } else {
      return this.request("/compare_files", {
        method: "POST",
      });
    }
  },

  async exportToExcel(data) {
    if (isDevelopment && window.pywebview) {
      return await window.pywebview.api.export_to_excel(data);
    } else {
      try {
        const url = isDevelopment
          ? `${API_BASE_URL}/export_excel`
          : "/api/export_excel";

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ data }),
        });

        if (response.ok) {
          const blob = await response.blob();
          const downloadUrl = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.style.display = "none";
          a.href = downloadUrl;
          a.download = `network_data_export_${new Date()
            .toISOString()
            .slice(0, 10)}.xlsx`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(downloadUrl);
          document.body.removeChild(a);
          return { status: "success", message: "File downloaded successfully" };
        } else {
          throw new Error("Export failed");
        }
      } catch (error) {
        throw new Error(`Export error: ${error.message}`);
      }
    }
  },

  async generateChartData(data, filterBy) {
    if (isDevelopment && window.pywebview) {
      return await window.pywebview.api.generate_chart_data(data, filterBy);
    } else {
      return this.request("/generate_chart", {
        method: "POST",
        body: JSON.stringify({ data, filter_by: filterBy }),
      });
    }
  },

  async getProgressChart(sessionId) {
    if (isDevelopment && sessionId) {
      return this.request(`/progress_chart/${sessionId}`);
    } else {
      return null;
    }
  },
};

// --- Helper Components ---

// File Upload Component
const FileUploadComponent = ({
  onFileUpload,
  disabled,
  uploadedFile,
  onFileRemove,
}) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFileUpload(e.dataTransfer.files[0]);
      }
    },
    [onFileUpload]
  );

  const handleChange = useCallback(
    (e) => {
      e.preventDefault();
      if (e.target.files && e.target.files[0]) {
        onFileUpload(e.target.files[0]);
      }
    },
    [onFileUpload]
  );

  return (
    <div className="file-upload-container">
      {!uploadedFile ? (
        <div
          className={`file-upload-area ${dragActive ? "drag-active" : ""} ${
            disabled ? "disabled" : ""
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            accept=".csv"
            onChange={handleChange}
            disabled={disabled}
            className="file-input"
          />
          <label htmlFor="file-upload" className="file-upload-label">
            <CloudUpload size={48} className="upload-icon" />
            <div className="upload-text">
              <h4>Upload CSV File</h4>
              <p>
                Drag and drop your device list CSV file here, or click to browse
              </p>
              <small>Supported format: CSV files only</small>
            </div>
          </label>
        </div>
      ) : (
        <div className="uploaded-file-info">
          <div className="file-success">
            <File size={24} className="file-icon" />
            <div className="file-details">
              <h4>File Uploaded Successfully</h4>
              <p>{uploadedFile.name}</p>
              <small>{uploadedFile.deviceCount} devices found</small>
            </div>
            {onFileRemove && (
              <button
                className="remove-file-btn"
                onClick={onFileRemove}
                disabled={disabled}
                aria-label="Remove file"
              >
                <X size={16} />
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const FileUploadComponentWithWarnings = ({
  onFileUpload,
  disabled,
  uploadedFile,
  onFileRemove,
}) => {
  return (
    <div className="file-upload-wrapper">
      <FileUploadComponent
        onFileUpload={onFileUpload}
        disabled={disabled}
        uploadedFile={uploadedFile}
        onFileRemove={onFileRemove}
      />
      {uploadedFile?.warnings && uploadedFile.warnings.length > 0 && (
        <div className="upload-warnings">
          <AlertTriangle size={16} />
          <span>{uploadedFile.warnings.length} warnings found in CSV</span>
        </div>
      )}
    </div>
  );
};

// Loading Overlay Component with Progress Bar
const LoadingOverlay = ({ message, isVisible, apiStatus, progress }) => {
  if (!isVisible) return null;

  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <Loader2 size={48} className="loading-spinner-icon animate-spin" />
        <h2>Network Data App</h2>
        <p>{message}</p>
        {progress && (
          <div className="progress-container">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress.percentage || 0}%` }}
              ></div>
            </div>
            <div className="progress-text">
              {progress.completed || 0} / {progress.total || 0} devices (
              {progress.percentage || 0}%)
            </div>
          </div>
        )}
        <div className="api-status-indicator">
          <div className={`status-dot ${apiStatus}`}></div>
          <span className="status-text">
            {apiStatus === "connecting" && "Connecting to Backend..."}
            {apiStatus === "ready" && "Backend Connected"}
            {apiStatus === "error" && "Connection Failed"}
          </span>
        </div>
        <div className="loading-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
};

// Alert component
const Alert = ({ info, onClose }) => {
  if (!info) return null;
  return (
    <div className="alert-dialog-backdrop" onClick={onClose}>
      <div
        className={`alert-dialog alert-${info.type}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="alert-header">
          <h3>
            {info.type === "error" && <XCircle size={20} />}
            {info.type === "success" && <CheckCircle size={20} />}
            {info.type === "warning" && <AlertTriangle size={20} />}
            {info.type === "info" && <TrendingUp size={20} />}
            Notification
          </h3>
          <button
            className="alert-close-btn"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={24} />
          </button>
        </div>
        <p className="alert-message">{info.message}</p>
        <button className="alert-ok-button" onClick={onClose}>
          OK
        </button>
      </div>
    </div>
  );
};

// Modal for showing detailed data
const DetailModal = ({ data, onClose, title = "Device Command Output" }) => {
  if (!data) return null;
  const formattedData =
    typeof data === "object" ? JSON.stringify(data, null, 2) : String(data);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content large-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <pre className="detail-content">{formattedData}</pre>
        </div>
      </div>
    </div>
  );
};

// File Comparison Modal
const ComparisonModal = ({ comparisonData, onClose }) => {
  const [searchTerm, setSearchTerm] = useState("");

  if (!comparisonData) return null;

  const filteredData =
    comparisonData.data?.filter(
      (device) =>
        device.ip_mgmt?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        device.nama_sw?.toLowerCase().includes(searchTerm.toLowerCase())
    ) || [];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content xl-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>
            <GitCompare size={24} />
            Configuration Comparison
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <div className="comparison-summary">
            <p>
              Comparing: <strong>{comparisonData.file1}</strong> vs{" "}
              <strong>{comparisonData.file2}</strong>
            </p>
            <p>
              Total devices compared:{" "}
              <strong>{comparisonData.total_devices_compared}</strong>
            </p>
          </div>

          <div className="search-box">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search devices..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="comparison-results">
            {filteredData.map((device, index) => (
              <div
                key={index}
                className={`comparison-item ${device.status.toLowerCase()}`}
              >
                <div className="device-header">
                  <h4>
                    {device.ip_mgmt} - {device.nama_sw}
                  </h4>
                  <span
                    className={`status-badge ${device.status.toLowerCase()}`}
                  >
                    {device.status}
                  </span>
                </div>
                <div className="device-details">
                  <p>{device.details}</p>
                  {device.changes && device.changes.length > 0 && (
                    <div className="changes-list">
                      <h5>Changes:</h5>
                      <ul>
                        {device.changes.slice(0, 10).map((change, idx) => (
                          <li
                            key={idx}
                            className={
                              change.startsWith("ADDED:")
                                ? "added"
                                : change.startsWith("REMOVED:")
                                ? "removed"
                                : "modified"
                            }
                          >
                            {change}
                          </li>
                        ))}
                        {device.changes.length > 10 && (
                          <li className="more-changes">
                            ... and {device.changes.length - 10} more changes
                          </li>
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Progress Component
const ProgressBar = ({ progress, showDetails = true }) => {
  if (!progress) return null;

  const percentage = progress.percentage || 0;
  const successRate =
    progress.total > 0 ? (progress.successful / progress.total) * 100 : 0;

  return (
    <div className="progress-section">
      <div className="progress-header">
        <h4>Processing Progress</h4>
        <span className="progress-percentage">{percentage.toFixed(1)}%</span>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>

      {showDetails && (
        <div className="progress-details">
          <div className="progress-stats">
            <div className="stat">
              <CheckCircle size={16} className="success" />
              <span>Success: {progress.successful || 0}</span>
            </div>
            <div className="stat">
              <XCircle size={16} className="error" />
              <span>Failed: {progress.failed || 0}</span>
            </div>
            <div className="stat">
              <Clock size={16} className="pending" />
              <span>
                Remaining: {(progress.total || 0) - (progress.completed || 0)}
              </span>
            </div>
          </div>
          <div className="success-rate">
            Success Rate: {successRate.toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
};

// Device Status Icon Component
const DeviceStatusIcon = ({ status, size = 16 }) => {
  const statusIcons = {
    [DEVICE_STATUS.SUCCESS]: (
      <CheckCircle size={size} className="status-success" />
    ),
    [DEVICE_STATUS.FAILED]: <XCircle size={size} className="status-failed" />,
    [DEVICE_STATUS.CONNECTING]: (
      <Loader2 size={size} className="status-connecting animate-spin" />
    ),
    [DEVICE_STATUS.RETRYING]: (
      <RefreshCw size={size} className="status-retrying animate-spin" />
    ),
    [DEVICE_STATUS.PENDING]: <Clock size={size} className="status-pending" />,
    [DEVICE_STATUS.STOPPED]: (
      <StopCircle size={size} className="status-stopped" />
    ),
  };

  return (
    statusIcons[status] || <Clock size={size} className="status-unknown" />
  );
};

// Output Files Manager
const OutputFilesModal = ({ isOpen, onClose, onCompareSelect }) => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  const loadFiles = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.request("/output_files");
      if (response.status === "success") {
        setFiles(response.data || []);
      }
    } catch (error) {
      console.error("Error loading files:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadFiles();
    }
  }, [isOpen, loadFiles]);

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete ${filename}?`)) return;

    try {
      const response = await api.request(`/output_files/${filename}`, {
        method: "DELETE",
      });
      if (response.status === "success") {
        loadFiles();
      }
    } catch (error) {
      alert(`Error deleting file: ${error.message}`);
    }
  };

  const handleDownload = (filename) => {
    const url = isDevelopment
      ? `${API_BASE_URL}/output_files/${filename}`
      : `/api/output_files/${filename}`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const toggleFileSelection = (filepath) => {
    setSelectedFiles((prev) => {
      const newSelection = prev.includes(filepath)
        ? prev.filter((f) => f !== filepath)
        : [...prev, filepath];
      return newSelection.slice(-2);
    });
  };

  const handleCompare = () => {
    if (selectedFiles.length === 2) {
      onCompareSelect(selectedFiles);
      onClose();
    }
  };

  const filteredFiles = files.filter((file) =>
    file.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content xl-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>
            <FolderOpen size={24} />
            Output Files Manager
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <div className="file-manager-controls">
            <div className="search-box">
              <Search size={16} />
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            {selectedFiles.length === 2 && (
              <button className="compare-button" onClick={handleCompare}>
                <GitCompare size={16} />
                Compare Selected Files
              </button>
            )}
            <button className="refresh-button" onClick={loadFiles}>
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>

          {loading ? (
            <div className="loading-container">
              <Loader2 size={32} className="animate-spin" />
              <p>Loading files...</p>
            </div>
          ) : (
            <div className="files-grid">
              {filteredFiles.length === 0 ? (
                <div className="empty-state">
                  <FolderOpen size={48} />
                  <p>No output files found</p>
                </div>
              ) : (
                filteredFiles.map((file) => (
                  <div
                    key={file.filename}
                    className={`file-card ${
                      selectedFiles.includes(file.filepath) ? "selected" : ""
                    }`}
                    onClick={() => toggleFileSelection(file.filepath)}
                  >
                    <div className="file-icon">
                      <File size={32} />
                    </div>
                    <div className="file-info">
                      <h4>{file.filename}</h4>
                      <p>Size: {formatFileSize(file.size)}</p>
                      <p>
                        Modified: {new Date(file.modified).toLocaleString()}
                      </p>
                    </div>
                    <div className="file-actions">
                      <button
                        className="action-btn download"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownload(file.filename);
                        }}
                        title="Download"
                      >
                        <Download size={16} />
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(file.filename);
                        }}
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Add Logs Viewer Component
const LogsViewer = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [filterLevel, setFilterLevel] = useState("ALL");

  const loadLogs = useCallback(async () => {
    try {
      const response = await api.request("/logs");
      if (response.status === "success") {
        setLogs(response.data || []);
      }
    } catch (error) {
      console.error("Error loading logs:", error);
    }
  }, []);

  useEffect(() => {
    if (isOpen && autoRefresh) {
      loadLogs();
      const interval = setInterval(loadLogs, 2000);
      return () => clearInterval(interval);
    }
  }, [isOpen, autoRefresh, loadLogs]);

  const handleClearLogs = async () => {
    if (!window.confirm("Are you sure you want to clear all logs?")) return;

    try {
      await api.request("/logs/clear", { method: "POST" });
      setLogs([]);
    } catch (error) {
      console.error("Error clearing logs:", error);
    }
  };

  const filteredLogs = useMemo(() => {
    if (filterLevel === "ALL") return logs;
    return logs.filter((log) => log.level === filterLevel);
  }, [logs, filterLevel]);

  const getLogLevelClass = (level) => {
    switch (level) {
      case "ERROR":
        return "log-error";
      case "WARNING":
        return "log-warning";
      case "INFO":
        return "log-info";
      default:
        return "log-debug";
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content xl-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>
            <Terminal size={24} />
            System Logs
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <div className="logs-controls">
            <div className="filter-group">
              <label>Level:</label>
              <select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value)}
              >
                <option value="ALL">All Levels</option>
                <option value="ERROR">Error</option>
                <option value="WARNING">Warning</option>
                <option value="INFO">Info</option>
                <option value="DEBUG">Debug</option>
              </select>
            </div>
            <label className="auto-refresh">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
            <button className="clear-logs-btn" onClick={handleClearLogs}>
              <Trash2 size={16} />
              Clear Logs
            </button>
          </div>

          <div className="logs-container">
            {filteredLogs.length === 0 ? (
              <div className="empty-logs">
                <Terminal size={48} />
                <p>No logs to display</p>
              </div>
            ) : (
              <div className="logs-list">
                {filteredLogs.map((log, index) => (
                  <div
                    key={index}
                    className={`log-entry ${getLogLevelClass(log.level)}`}
                  >
                    <span className="log-timestamp">{log.timestamp}</span>
                    <span className="log-level">[{log.level}]</span>
                    <span className="log-module">{log.module}:</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Main App Component ---
function App() {
  // State management
  const [message, setMessage] = useState(MESSAGES.INITIALIZING);
  const [status, setStatus] = useState(STATUS.LOADING);
  const [results, setResults] = useState([]);
  const [filteredResults, setFilteredResults] = useState([]);
  const [credentials, setCredentials] = useState({
    username: "",
    password: "",
  });
  const [detailData, setDetailData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [chartFilter, setChartFilter] = useState("model_sw");
  const [alertInfo, setAlertInfo] = useState(null);
  const [isApiReady, setIsApiReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [systemInfo, setSystemInfo] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [apiStatus, setApiStatus] = useState("connecting");

  // File upload state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFilePath, setUploadedFilePath] = useState("");

  // Processing state
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [canRetry, setCanRetry] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [showFileManager, setShowFileManager] = useState(false);
  const [showLogsViewer, setShowLogsViewer] = useState(false);
  const [comparisonFiles, setComparisonFiles] = useState(null);

  // Filtering state
  const [filterType, setFilterType] = useState("all");
  const [filterValue, setFilterValue] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  // Dark mode
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem("darkMode");
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(isDarkMode));
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  const toggleDarkMode = useCallback(() => {
    setIsDarkMode((prev) => !prev);
  }, []);

  // Memoized derived state for performance
  const hasResults = useMemo(() => results.length > 0, [results]);
  const displayResults = useMemo(() => {
    let filtered = filteredResults.length > 0 ? filteredResults : results;

    if (searchTerm) {
      filtered = filtered.filter(
        (device) =>
          device.ip_mgmt?.toLowerCase().includes(searchTerm.toLowerCase()) ||
          device.nama_sw?.toLowerCase().includes(searchTerm.toLowerCase()) ||
          device.model_sw?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return filtered;
  }, [filteredResults, results, searchTerm]);

  const successCount = useMemo(
    () => displayResults.filter((r) => r.status === "Success").length,
    [displayResults]
  );
  const failureCount = useMemo(
    () => displayResults.filter((r) => r.status === "Failed").length,
    [displayResults]
  );

  // Callback for showing alerts
  const showAlert = useCallback((message, type = ALERT_TYPES.INFO) => {
    setAlertInfo({ message, type });
  }, []);

  // File upload handler
  const handleFileUpload = useCallback(
    async (file) => {
      if (!file || file.type !== "text/csv") {
        showAlert("Please select a valid CSV file", ALERT_TYPES.ERROR);
        return;
      }

      if (file.size > 16 * 1024 * 1024) {
        showAlert(
          "File size too large. Maximum size is 16MB",
          ALERT_TYPES.ERROR
        );
        return;
      }

      try {
        setMessage("Uploading file...");
        const result = await api.uploadCsvFile(file);

        if (result.status === "success") {
          setUploadedFile({
            name: file.name,
            deviceCount: result.device_count,
            warnings: result.warnings || [],
          });
          setUploadedFilePath(result.filepath);

          let message = result.message;
          if (result.warnings && result.warnings.length > 0) {
            message += ` (${result.warnings.length} warnings found)`;
          }

          showAlert(message, ALERT_TYPES.SUCCESS);
          setMessage(MESSAGES.API_READY);
        } else {
          let errorMessage = result.message || "File upload failed";
          if (result.errors && result.errors.length > 0) {
            errorMessage = result.errors.join("\n");
          }
          showAlert(errorMessage, ALERT_TYPES.ERROR);
        }
      } catch (error) {
        console.error("File upload error:", error);
        showAlert(`Upload failed: ${error.message}`, ALERT_TYPES.ERROR);
      }
    },
    [showAlert]
  );

  // Remove uploaded file
  const handleFileRemove = useCallback(() => {
    setUploadedFile(null);
    setUploadedFilePath("");
    setMessage(MESSAGES.AWAITING_FILE);
  }, []);

  // Callback to handle responses from backend
  const handleBackendResponse = useCallback(
    (response) => {
      console.log("Response from backend:", response);
      setIsProcessing(false);
      setProgress(null);

      if (!response) {
        setMessage("Received an empty response from backend.");
        setStatus(STATUS.ERROR);
        showAlert(
          "Received an empty response from backend.",
          ALERT_TYPES.ERROR
        );
        return;
      }

      setMessage(response.message || MESSAGES.PROCESS_FINISHED);
      setStatus(response.status || STATUS.IDLE);

      if (response.data) {
        setResults(response.data);
        setFilteredResults([]);
        setCanRetry(response.data.some((device) => device.status === "Failed"));
      }

      if (
        response.status === STATUS.ERROR ||
        response.status === STATUS.SUCCESS
      ) {
        showAlert(response.message, response.status);
      }
    },
    [showAlert]
  );

  // Effect to check processing status
  useEffect(() => {
    let statusInterval;

    if (isProcessing && currentSessionId) {
      statusInterval = setInterval(async () => {
        try {
          const statusResponse = await api.getProcessingStatus(
            currentSessionId
          );

          if (statusResponse && statusResponse.status !== "processing") {
            handleBackendResponse(statusResponse);
            setCurrentSessionId(null);
          } else if (statusResponse && statusResponse.progress) {
            setProgress(statusResponse.progress);
            setMessage(statusResponse.message);
          }
        } catch (error) {
          console.error("Error checking processing status:", error);
          setIsProcessing(false);
          setCurrentSessionId(null);
          setProgress(null);
          showAlert("Error checking processing status", ALERT_TYPES.ERROR);
        }
      }, 2000); // Check every 2 seconds
    }

    return () => {
      if (statusInterval) {
        clearInterval(statusInterval);
      }
    };
  }, [isProcessing, currentSessionId, handleBackendResponse, showAlert]);

  // Effect to set up API and listeners on component mount
  useEffect(() => {
    const initializeApi = async () => {
      try {
        setApiStatus("ready");
        setMessage("Backend connected successfully");

        // Test connection
        try {
          const sysInfo = await api.getSystemInfo();
          if (sysInfo && sysInfo.status === "success") {
            setSystemInfo(sysInfo.data);
            console.log("System info loaded successfully:", sysInfo.data);
          }
        } catch (error) {
          console.error("Failed to connect to backend:", error);
          setApiStatus("error");
          setMessage("Failed to connect to backend");
          showAlert(
            `Cannot connect to backend server. ${
              isDevelopment
                ? "Make sure Flask server is running on localhost:5000"
                : "Backend service unavailable"
            }`,
            ALERT_TYPES.ERROR
          );
        }

        setTimeout(() => {
          setIsApiReady(true);
          setIsInitializing(false);
          setMessage(
            isDevelopment && !window.pywebview
              ? MESSAGES.AWAITING_FILE
              : MESSAGES.API_READY
          );
          setStatus(STATUS.IDLE);
        }, 1000);

        // Set up PyWebView listeners if in development mode
        if (isDevelopment && window.pywebview) {
          // Expose frontend functions to be called from Python
          window.handlePythonResponse = handleBackendResponse;
          window.setDetailData = (data) => setDetailData(data);
          window.setAlertInfo = (alert) => showAlert(alert.message, alert.type);
        }
      } catch (error) {
        console.error("Error initializing API:", error);
        setApiStatus("error");
        setIsInitializing(false);
        showAlert("Failed to initialize backend connection", ALERT_TYPES.ERROR);
      }
    };

    initializeApi();

    return () => {
      // Cleanup PyWebView listeners
      if (isDevelopment && window.pywebview) {
        delete window.handlePythonResponse;
        delete window.setDetailData;
        delete window.setAlertInfo;
      }
    };
  }, [handleBackendResponse, showAlert]);

  // Effect to generate chart data when results or filter change
  useEffect(() => {
    const generateChart = async () => {
      if (!hasResults || !isApiReady) {
        setChartData(null);
        return;
      }

      try {
        const dataToChart =
          displayResults.length > 0 ? displayResults : results;
        const chartResponse = await api.generateChartData(
          dataToChart,
          chartFilter
        );
        if (chartResponse?.status === "success") {
          setChartData(chartResponse.data);
        } else {
          setChartData(null);
          console.error("Chart generation failed:", chartResponse?.message);
        }
      } catch (e) {
        console.error("Error fetching chart data:", e);
        setChartData(null);
      }
    };

    generateChart();
  }, [results, displayResults, chartFilter, hasResults, isApiReady]);

  // Handler to start the main process
  const handleRunScript = useCallback(
    async (retryFailedOnly = false) => {
      if (!isApiReady) {
        return showAlert(
          "API is not ready. Please restart the application.",
          ALERT_TYPES.ERROR
        );
      }

      // Check if we need a file upload (production mode) or can use file dialog (development)
      if (!isDevelopment || !window.pywebview) {
        if (!uploadedFilePath && !retryFailedOnly) {
          showAlert("Please upload a CSV file first", ALERT_TYPES.ERROR);
          return;
        }
      }

      if (!credentials.username.trim() || !credentials.password.trim()) {
        showAlert(
          "Warning: Username or password is empty. This may cause connection failures.",
          ALERT_TYPES.WARNING
        );
      }

      setMessage(
        retryFailedOnly
          ? "Starting retry process..."
          : isDevelopment && window.pywebview
          ? MESSAGES.AWAITING_FILE
          : "Starting processing..."
      );
      setStatus(STATUS.LOADING);
      setIsProcessing(true);
      if (!retryFailedOnly) {
        setResults([]);
        setFilteredResults([]);
      }
      setChartData(null);
      setAlertInfo(null);
      setProgress(null);

      try {
        let initialResponse;

        if (retryFailedOnly) {
          initialResponse = await api.retryFailedDevices(
            credentials.username,
            credentials.password,
            results
          );
        } else {
          initialResponse = await api.processDevicesFromFile(
            credentials.username,
            credentials.password,
            uploadedFilePath,
            retryFailedOnly
          );
        }

        if (initialResponse) {
          setMessage(initialResponse.message);
          setStatus(initialResponse.status);

          if (initialResponse.session_id) {
            setCurrentSessionId(initialResponse.session_id);
            if (initialResponse.total_devices) {
              setProgress({
                total: initialResponse.total_devices,
                completed: 0,
                successful: 0,
                failed: 0,
                percentage: 0,
              });
            }
          }

          if (
            initialResponse.status === "error" ||
            initialResponse.status === "info"
          ) {
            setIsProcessing(false);
            setProgress(null);
            showAlert(initialResponse.message, initialResponse.status);
          }
        }
      } catch (e) {
        console.error("Error starting script:", e);
        const errorMessage = `An error occurred: ${e.message || e}`;
        showAlert(errorMessage, ALERT_TYPES.ERROR);
        setMessage(errorMessage);
        setStatus(STATUS.ERROR);
        setIsProcessing(false);
        setProgress(null);
      }
    },
    [isApiReady, credentials, showAlert, results, uploadedFilePath]
  );

  // Handler to stop processing
  const handleStopProcessing = useCallback(async () => {
    if (!currentSessionId) return;

    try {
      const response = await api.stopProcessing(currentSessionId);
      if (response && response.status === "success") {
        showAlert("Stop request sent successfully", ALERT_TYPES.INFO);
        setStatus(STATUS.STOPPED);
        setMessage("Stopping processing...");
      }
    } catch (e) {
      showAlert(`Error stopping process: ${e.message || e}`, ALERT_TYPES.ERROR);
    }
  }, [currentSessionId, showAlert]);

  // Handler for filtering results
  const handleFilterChange = useCallback(
    async (newFilterType, newFilterValue) => {
      setFilterType(newFilterType);
      setFilterValue(newFilterValue);

      if (newFilterType === "all" || !newFilterValue) {
        setFilteredResults([]);
        return;
      }

      try {
        const response = await api.filterResults(
          results,
          newFilterType,
          newFilterValue
        );
        if (response && response.status === "success") {
          setFilteredResults(response.data);
        }
      } catch (e) {
        console.error("Error filtering results:", e);
        showAlert("Error applying filter", ALERT_TYPES.ERROR);
      }
    },
    [results, showAlert]
  );

  // Handler for file comparison
  const handleCompareFiles = useCallback(
    async (files = null) => {
      try {
        if (!files) {
          setShowFileManager(true);
          return;
        }

        const response = await api.request("/compare_files", {
          method: "POST",
          body: JSON.stringify({
            file1: files[0],
            file2: files[1],
          }),
        });

        if (response && response.status === "success") {
          setComparisonData(response);
        } else {
          showAlert(
            response?.message || "Error comparing files",
            ALERT_TYPES.ERROR
          );
        }
      } catch (e) {
        showAlert(`Comparison error: ${e.message || e}`, ALERT_TYPES.ERROR);
      }
    },
    [showAlert]
  );

  // Handle file comparison selection from file manager
  const handleCompareSelect = useCallback(
    (selectedFiles) => {
      setComparisonFiles(selectedFiles);
      handleCompareFiles(selectedFiles);
    },
    [handleCompareFiles]
  );

  // Handler for exporting data to Excel
  const handleExport = useCallback(async () => {
    if (!hasResults) return showAlert("No data to export.", ALERT_TYPES.INFO);
    if (isProcessing) return;

    try {
      const dataToExport = displayResults.length > 0 ? displayResults : results;
      const response = await api.exportToExcel(dataToExport);
      if (response && response.status === "success") {
        showAlert(response.message, ALERT_TYPES.SUCCESS);
      }
    } catch (e) {
      showAlert(`Export error: ${e.message || e}`, ALERT_TYPES.ERROR);
    }
  }, [hasResults, isProcessing, displayResults, results, showAlert]);

  return (
    <div className="app-container">
      <LoadingOverlay
        message={message}
        isVisible={isInitializing}
        apiStatus={apiStatus}
        progress={progress}
      />

      <Alert info={alertInfo} onClose={() => setAlertInfo(null)} />
      <DetailModal data={detailData} onClose={() => setDetailData(null)} />
      <ComparisonModal
        comparisonData={comparisonData}
        onClose={() => setComparisonData(null)}
      />
      <OutputFilesModal
        isOpen={showFileManager}
        onClose={() => setShowFileManager(false)}
        onCompareSelect={handleCompareSelect}
      />
      <LogsViewer
        isOpen={showLogsViewer}
        onClose={() => setShowLogsViewer(false)}
      />

      <header className="app-header">
        <div className="logo">
          <Server size={40} className="logo-icon" />
          <div>
            <h1>Network Data App</h1>
            <p>Automated data collection from network devices</p>
          </div>
        </div>
        <div className="header-controls">
          {systemInfo && (
            <div className="system-info">
              <small>Version: {systemInfo.version || "2.0.0"}</small>
              <div className={`api-connection-status ${apiStatus}`}>
                <div className="status-indicator"></div>
                <span>
                  {apiStatus === "ready" ? "Connected" : "Disconnected"}
                  {isDevelopment && " (Dev Mode)"}
                </span>
              </div>
              {systemInfo.features && (
                <div className="features-list">
                  <small>Features: {systemInfo.features.join(", ")}</small>
                </div>
              )}
            </div>
          )}
          <div className="header-buttons">
            <button
              className="header-btn"
              onClick={() => setShowFileManager(true)}
              title="Output Files"
            >
              <FolderOpen size={20} />
            </button>
            <button
              className="header-btn"
              onClick={() => setShowLogsViewer(true)}
              title="System Logs"
            >
              <Terminal size={20} />
            </button>
            <button
              className="dark-mode-toggle"
              onClick={toggleDarkMode}
              aria-label="Toggle dark mode"
            >
              {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="setup-grid">
          {/* File Upload Section */}
          {(!isDevelopment || !window.pywebview) && (
            <div className="card">
              <div className="card-header">
                <Upload size={20} />
                <h3>1. Upload Device List</h3>
              </div>
              <div className="card-content">
                <FileUploadComponentWithWarnings
                  onFileUpload={handleFileUpload}
                  disabled={isProcessing}
                  uploadedFile={uploadedFile}
                  onFileRemove={handleFileRemove}
                />
                <small className="upload-note">
                  CSV must contain: IP MGMT, Device Name (Nama SW), Serial
                  Number (SN), and Model (Model SW)
                </small>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <Key size={20} />
              <h3>
                {!isDevelopment || !window.pywebview
                  ? "2. SSH/TACACS Credentials"
                  : "1. SSH/TACACS Credentials"}
              </h3>
            </div>
            <div className="card-content">
              <input
                type="text"
                placeholder="Username (required)"
                value={credentials.username}
                onChange={(e) =>
                  setCredentials((p) => ({ ...p, username: e.target.value }))
                }
                className="credential-input"
                disabled={isProcessing}
                required
              />
              <input
                type="password"
                placeholder="Password (required)"
                value={credentials.password}
                onChange={(e) =>
                  setCredentials((p) => ({ ...p, password: e.target.value }))
                }
                className="credential-input"
                disabled={isProcessing}
                required
              />
              <small className="credential-note">
                <AlertCircle size={12} />
                Credentials are required for SSH/TACACS authentication
              </small>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <Play size={20} />
              <h3>
                {!isDevelopment || !window.pywebview
                  ? "3. Run Process"
                  : "2. Run Process"}
              </h3>
            </div>
            <div className="card-content">
              <p className="start-description">
                {isDevelopment && window.pywebview
                  ? "Click Start to open the file dialog and begin processing the devices from your CSV list."
                  : "Upload your device list CSV file and click Start to begin processing."}
              </p>
              <div className="button-group">
                <button
                  onClick={() => handleRunScript(false)}
                  disabled={
                    !isApiReady ||
                    isProcessing ||
                    !credentials.username ||
                    !credentials.password ||
                    ((!isDevelopment || !window.pywebview) && !uploadedFilePath)
                  }
                  className="run-button primary"
                >
                  <Activity size={18} />
                  {isProcessing ? MESSAGES.PROCESSING : "Start SSH Collection"}
                </button>

                {canRetry && (
                  <button
                    onClick={() => handleRunScript(true)}
                    disabled={
                      !isApiReady ||
                      isProcessing ||
                      !credentials.username ||
                      !credentials.password
                    }
                    className="run-button secondary"
                  >
                    <RefreshCw size={18} />
                    Retry Failed Devices
                  </button>
                )}

                {isProcessing && currentSessionId && (
                  <button
                    onClick={handleStopProcessing}
                    className="stop-button"
                  >
                    <StopCircle size={18} />
                    Stop Processing
                  </button>
                )}
              </div>

              {(!credentials.username || !credentials.password) && (
                <div className="validation-warning">
                  <Info size={16} />
                  <span>Please provide both username and password</span>
                </div>
              )}

              {!isApiReady && (
                <div className="api-status">
                  <Loader2 size={16} className="animate-spin" />
                  <span>Waiting for backend...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Progress Section */}
        {isProcessing && progress && (
          <div className="card">
            <div className="card-header">
              <TrendingUp size={20} />
              <h3>Processing Progress</h3>
            </div>
            <div className="card-content">
              <ProgressBar progress={progress} />
            </div>
          </div>
        )}

        {hasResults && (
          <>
            {/* Results Summary */}
            <div className="card results-summary">
              <div className="card-header">
                <h3>Processing Complete</h3>
                <div className="summary-stats">
                  <span>
                    Total: <strong>{displayResults.length}</strong>
                  </span>
                  <span className="success">
                    Success: <strong>{successCount}</strong>
                  </span>
                  <span className="failure">
                    Failed: <strong>{failureCount}</strong>
                  </span>
                  {filteredResults.length > 0 && (
                    <span className="filtered">
                      Filtered:{" "}
                      <strong>
                        {filteredResults.length}/{results.length}
                      </strong>
                    </span>
                  )}
                </div>
                <div className="action-buttons">
                  <button
                    onClick={handleExport}
                    className="action-button"
                    disabled={isProcessing}
                  >
                    <Download size={16} /> Export to Excel
                  </button>
                  <button
                    onClick={() => handleCompareFiles()}
                    className="action-button secondary"
                    disabled={isProcessing}
                  >
                    <GitCompare size={16} /> Compare Files
                  </button>
                  <button
                    onClick={() => setShowFileManager(true)}
                    className="action-button secondary"
                    disabled={isProcessing}
                  >
                    <FolderOpen size={16} /> Manage Files
                  </button>
                </div>
              </div>
            </div>

            {/* Filtering Section */}
            <div className="card">
              <div className="card-header">
                <Filter size={20} />
                <h3>Data Filtering</h3>
                <div className="filter-controls">
                  <div className="filter-group">
                    <label>Filter by:</label>
                    <select
                      value={filterType}
                      onChange={(e) =>
                        handleFilterChange(e.target.value, filterValue)
                      }
                      disabled={isProcessing}
                    >
                      <option value="all">All Devices</option>
                      <option value="status">Status</option>
                      <option value="model_sw">Model</option>
                      <option value="connection_status">
                        Connection Status
                      </option>
                    </select>
                  </div>

                  {filterType !== "all" && (
                    <div className="filter-group">
                      <label>Value:</label>
                      {filterType === "status" ? (
                        <select
                          value={filterValue}
                          onChange={(e) =>
                            handleFilterChange(filterType, e.target.value)
                          }
                          disabled={isProcessing}
                        >
                          <option value="">Select Status</option>
                          <option value="Success">Success</option>
                          <option value="Failed">Failed</option>
                        </select>
                      ) : filterType === "connection_status" ? (
                        <select
                          value={filterValue}
                          onChange={(e) =>
                            handleFilterChange(filterType, e.target.value)
                          }
                          disabled={isProcessing}
                        >
                          <option value="">Select Connection Status</option>
                          <option value="success">Success</option>
                          <option value="failed">Failed</option>
                          <option value="connecting">Connecting</option>
                          <option value="retrying">Retrying</option>
                          <option value="stopped">Stopped</option>
                        </select>
                      ) : (
                        <input
                          type="text"
                          placeholder="Enter filter value"
                          value={filterValue}
                          onChange={(e) =>
                            handleFilterChange(filterType, e.target.value)
                          }
                          disabled={isProcessing}
                        />
                      )}
                    </div>
                  )}

                  <div className="search-group">
                    <Search size={16} />
                    <input
                      type="text"
                      placeholder="Search devices..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      disabled={isProcessing}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Dashboard */}
            <div className="card">
              <div className="card-header">
                <BarChart2 size={20} />
                <h3>Dashboard</h3>
                <div className="filter-group">
                  <label htmlFor="chart-filter">Group by:</label>
                  <select
                    id="chart-filter"
                    value={chartFilter}
                    onChange={(e) => setChartFilter(e.target.value)}
                    disabled={isProcessing}
                  >
                    <option value="model_sw">Model</option>
                    <option value="status">Status</option>
                    <option value="connection_status">Connection Status</option>
                  </select>
                </div>
              </div>
              <div className="card-content">
                {chartData ? (
                  <Plot
                    data={chartData.data}
                    layout={{
                      ...chartData.layout,
                      autosize: true,
                      paper_bgcolor: "var(--secondary-bg)",
                      font: { color: "var(--text-color)" },
                    }}
                    style={{ width: "100%", height: "400px" }}
                    useResizeHandler
                    config={{ responsive: true, displaylogo: false }}
                  />
                ) : (
                  <p>Loading chart data...</p>
                )}
              </div>
            </div>

            {/* Results Details */}
            <div className="card">
              <div className="card-header">
                <FileText size={20} />
                <h3>Results Details</h3>
                {displayResults.length !== results.length && (
                  <span className="filter-indicator">
                    Showing {displayResults.length} of {results.length} devices
                  </span>
                )}
              </div>
              <div className="table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>IP</th>
                      <th>Hostname</th>
                      <th>Model</th>
                      <th>Device Type</th>
                      <th>Serial</th>
                      <th>Connection</th>
                      <th>Time (s)</th>
                      <th>Retries</th>
                      <th>Last Attempt</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayResults.map((device, index) => (
                      <tr
                        key={`${device.ip_mgmt}-${index}`}
                        className={`status-${device.status?.toLowerCase()}`}
                      >
                        <td className="status-cell">
                          <DeviceStatusIcon status={device.connection_status} />
                          <span
                            className={`badge ${
                              device.status === "Success"
                                ? "badge-success"
                                : "badge-danger"
                            }`}
                          >
                            {device.status}
                          </span>
                        </td>
                        <td className="ip-cell">{device.ip_mgmt || "N/A"}</td>
                        <td>{device.nama_sw || "N/A"}</td>
                        <td>{device.model_sw || "N/A"}</td>
                        <td className="device-type-cell">
                          <span className="device-type-badge">
                            {device.detected_device_type || "Unknown"}
                          </span>
                        </td>
                        <td>{device.sn || "N/A"}</td>
                        <td>
                          <span
                            className={`connection-status ${device.connection_status}`}
                          >
                            {device.connection_status || "Unknown"}
                          </span>
                        </td>
                        <td>{device.processing_time?.toFixed(2) ?? "N/A"}</td>
                        <td>
                          {device.retry_count > 0 ? (
                            <span className="retry-count">
                              <RefreshCw size={12} />
                              {device.retry_count}
                            </span>
                          ) : (
                            "0"
                          )}
                        </td>
                        <td className="timestamp-cell">
                          {device.last_attempt
                            ? new Date(device.last_attempt).toLocaleString()
                            : "N/A"}
                        </td>
                        <td className="details-cell">
                          {device.status === "Success" ? (
                            <button
                              className="view-button"
                              onClick={() => setDetailData(device.data)}
                            >
                              <Eye size={14} />
                              View
                            </button>
                          ) : (
                            <span className="error-text" title={device.error}>
                              {device.error
                                ? device.error.length > 50
                                  ? `${device.error.substring(0, 50)}...`
                                  : device.error
                                : "Unknown"}
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
