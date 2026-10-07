from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from plugins.moondream_vision.config_model import (
    MoondreamVisionConfig,
    default_config_path,
    load_config,
    save_config,
)
from sdk.plugin_host_context import PluginSettingsUIContext


class MoondreamVisionSettingsTab(QWidget):
    """设置 → 小工具：使用视觉适配器或本地 Moondream 识屏。"""

    def __init__(
        self, plg: PluginSettingsUIContext, plugin_root: Path, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._plg = plg
        self._root = plugin_root
        self._path = default_config_path(plugin_root)
        self._build()
        self._load_into_ui()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        hint = QLabel(
            "使用 mss 截屏，默认通过「AI 服务」中配置的视觉适配器识别屏幕。"
            "满足「屏幕相对上次识别有明显变化 / 鼠标移动 / （Windows）新开或切换前台窗口」时才会截屏送模型；"
            "并受下方「最短推理间隔」限制，避免过于频繁。\n\n"
            "自动识屏时按触发类型使用英文内置提示词（可在下方逐项覆盖）；多条件同时满足时优先级为："
            "屏幕差分 > 前台切换 > 新窗口 > 鼠标移动。"
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        box = QGroupBox("屏幕识别")
        fl = QFormLayout(box)
        self._enabled = QCheckBox("启用识屏（差分 / 鼠标 / 系统窗口事件触发）")
        fl.addRow(self._enabled)
        self._adapter = QComboBox()
        self._adapter.addItem("Vision adapter（AI 服务设置）", "vision")
        self._adapter.addItem("Moondream（本地）", "moondream")
        self._adapter.currentIndexChanged.connect(self._update_adapter_visibility)
        fl.addRow("适配器:", self._adapter)
        self._local_model_box = QGroupBox("Moondream 本地模型")
        model_fl = QFormLayout(self._local_model_box)
        local_hint = QLabel(
            "首次使用会下载模型到 Hugging Face 缓存。请先安装本地模型依赖；"
            "INT8 / INT4 需要 NVIDIA CUDA 和 bitsandbytes。"
        )
        local_hint.setWordWrap(True)
        model_fl.addRow(local_hint)
        self._model_id = QLineEdit()
        self._model_id.setPlaceholderText("vikhyatk/moondream2")
        model_fl.addRow("模型 ID:", self._model_id)
        self._revision = QLineEdit()
        self._revision.setPlaceholderText("可选，如 2025-01-09")
        model_fl.addRow("修订 revision:", self._revision)
        self._cache_dir = QLineEdit()
        self._cache_dir.setPlaceholderText("可选；留空用系统默认 HF 缓存")
        model_fl.addRow("缓存目录:", self._cache_dir)
        self._device = QComboBox()
        for label, data in (
            ("自动", "auto"),
            ("CUDA", "cuda"),
            ("Apple MPS", "mps"),
            ("CPU", "cpu"),
        ):
            self._device.addItem(label, data)
        model_fl.addRow("设备:", self._device)
        self._quantization = QComboBox()
        for label, data in (
            ("无（浮点）", "none"),
            ("INT8", "int8"),
            ("INT4（NF4）", "int4"),
        ):
            self._quantization.addItem(label, data)
        self._quantization.setToolTip(
            "INT8 / INT4 需 NVIDIA CUDA 与 bitsandbytes；与 Apple MPS / 纯 CPU 不兼容。"
            "视觉塔 vision 不参与量化（仍为 FP），以兼容 Moondream 自定义前向；仅语言部分减压显存。"
        )
        model_fl.addRow("权重量化:", self._quantization)
        fl.addRow(self._local_model_box)
        self._motion_poll = QDoubleSpinBox()
        self._motion_poll.setRange(0.12, 3.0)
        self._motion_poll.setSingleStep(0.05)
        self._motion_poll.setSuffix(" 秒")
        self._motion_poll.setToolTip("采样鼠标、窗口与缩略图差分的间隔；越小越灵敏、占用略高")
        fl.addRow("触发采样间隔:", self._motion_poll)
        self._diff_thr = QDoubleSpinBox()
        self._diff_thr.setRange(0.003, 0.35)
        self._diff_thr.setDecimals(3)
        self._diff_thr.setSingleStep(0.002)
        self._diff_thr.setToolTip("相对上次识别成功的缩略图，变化像素占比阈值")
        fl.addRow("屏幕差分阈值:", self._diff_thr)
        self._mouse_pct = QDoubleSpinBox()
        self._mouse_pct.setRange(0.02, 25.0)
        self._mouse_pct.setDecimals(2)
        self._mouse_pct.setSingleStep(0.05)
        self._mouse_pct.setSuffix(" %")
        self._mouse_pct.setToolTip(
            "相对当前「显示器索引」对应画面的宽/高较大一边；移动直线距离超过该比例视为活动"
        )
        fl.addRow("鼠标移动阈值(% 屏):", self._mouse_pct)
        self._interval = QDoubleSpinBox()
        self._interval.setRange(5.0, 600.0)
        self._interval.setSingleStep(1.0)
        self._interval.setSuffix(" 秒")
        self._interval.setToolTip("两次送模型推理之间的最短间隔（满足触发条件后仍可能排队等待）")
        fl.addRow("最短推理间隔:", self._interval)
        self._monitor = QSpinBox()
        self._monitor.setRange(0, 16)
        self._monitor.setToolTip("mss：0=所有显示器合成；1 通常为第一块物理屏")
        fl.addRow("显示器索引:", self._monitor)
        self._infer_max_side = QSpinBox()
        self._infer_max_side.setRange(0, 8192)
        self._infer_max_side.setSingleStep(128)
        self._infer_max_side.setSpecialValueText("不缩放")
        self._infer_max_side.setToolTip(
            "送入 Moondream 前将截图较长边缩到此像素；0=不缩放。\n"
            "4K 全屏不缩在 CPU 上可能单次 query 需数分钟；默认 1280 可明显加速。"
        )
        fl.addRow("推理输入最长边 (px):", self._infer_max_side)

        pq = QGroupBox("英文提示词（按触发类型，可留空用内置）")
        pql = QFormLayout(pq)
        self._q_screen_diff = QTextEdit()
        self._q_screen_diff.setMaximumHeight(72)
        self._q_screen_diff.setPlaceholderText(
            "screen_diff: screen thumbnail changed a lot since last successful capture"
        )
        pql.addRow("屏幕差分 · screen_diff:", self._q_screen_diff)
        self._q_foreground = QTextEdit()
        self._q_foreground.setMaximumHeight(72)
        self._q_foreground.setPlaceholderText(
            "foreground: focused window changed (Windows)"
        )
        pql.addRow("前台切换 · foreground:", self._q_foreground)
        self._q_new_window = QTextEdit()
        self._q_new_window.setMaximumHeight(72)
        self._q_new_window.setPlaceholderText(
            "new_window: new top-level window opened"
        )
        pql.addRow("新窗口 · new_window:", self._q_new_window)
        self._q_mouse = QTextEdit()
        self._q_mouse.setMaximumHeight(72)
        self._q_mouse.setPlaceholderText(
            "mouse: user moved mouse beyond threshold"
        )
        pql.addRow("鼠标移动 · mouse:", self._q_mouse)
        self._question = QTextEdit()
        self._question.setMaximumHeight(60)
        self._question.setPlaceholderText(
            "Legacy: one English prompt for ALL triggers only if the four fields above are all empty"
        )
        pql.addRow("统一提问（可选）:", self._question)
        fl.addRow(pq)

        self._prefix = QLineEdit()
        self._prefix.setPlaceholderText("发到聊天里的前缀")
        fl.addRow("消息前缀:", self._prefix)

        row = QHBoxLayout()
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self._on_save)
        row.addWidget(save_btn)
        row.addStretch(1)
        fl.addRow(row)
        lay.addWidget(box)
        lay.addStretch(1)

    def _quant_index(self, mode: str) -> int:
        m = (mode or "none").strip().lower()
        for i in range(self._quantization.count()):
            if str(self._quantization.itemData(i)).lower() == m:
                return i
        return 0

    def _device_index(self, device: str) -> int:
        d = (device or "auto").strip().lower()
        for i in range(self._device.count()):
            if str(self._device.itemData(i)).lower() == d:
                return i
        return 0

    def _load_into_ui(self) -> None:
        c = load_config(self._path)
        self._enabled.setChecked(c.enabled)
        self._adapter.setCurrentIndex(max(0, self._adapter.findData(c.adapter)))
        self._update_adapter_visibility()
        self._model_id.setText(c.model_id)
        self._revision.setText(c.revision)
        self._cache_dir.setText(c.cache_dir)
        self._device.setCurrentIndex(self._device_index(c.device))
        self._quantization.setCurrentIndex(self._quant_index(c.quantization))
        self._motion_poll.setValue(c.motion_poll_sec)
        self._diff_thr.setValue(c.diff_threshold)
        self._mouse_pct.setValue(c.mouse_move_percent)
        self._interval.setValue(c.interval_sec)
        self._monitor.setValue(c.monitor_index)
        self._infer_max_side.setValue(c.infer_max_side)
        self._q_screen_diff.setPlainText(c.question_screen_diff)
        self._q_foreground.setPlainText(c.question_foreground)
        self._q_new_window.setPlainText(c.question_new_window)
        self._q_mouse.setPlainText(c.question_mouse)
        self._question.setPlainText(c.question)
        self._prefix.setText(c.message_prefix)

    def _read_from_ui(self) -> MoondreamVisionConfig:
        c = MoondreamVisionConfig(
            enabled=self._enabled.isChecked(),
            adapter=str(self._adapter.currentData() or "vision"),
            model_id=self._model_id.text().strip() or "vikhyatk/moondream2",
            revision=self._revision.text().strip(),
            cache_dir=self._cache_dir.text().strip(),
            device=str(self._device.currentData() or "auto"),
            quantization=str(self._quantization.currentData() or "none"),
            motion_poll_sec=float(self._motion_poll.value()),
            diff_threshold=float(self._diff_thr.value()),
            mouse_move_percent=float(self._mouse_pct.value()),
            interval_sec=float(self._interval.value()),
            monitor_index=int(self._monitor.value()),
            infer_max_side=int(self._infer_max_side.value()),
            question=self._question.toPlainText().strip(),
            question_screen_diff=self._q_screen_diff.toPlainText().strip(),
            question_foreground=self._q_foreground.toPlainText().strip(),
            question_new_window=self._q_new_window.toPlainText().strip(),
            question_mouse=self._q_mouse.toPlainText().strip(),
            message_prefix=self._prefix.text(),
        )
        c.clamp()
        return c

    def _update_adapter_visibility(self) -> None:
        self._local_model_box.setVisible(self._adapter.currentData() == "moondream")

    def _on_save(self) -> None:
        c = self._read_from_ui()
        save_config(self._path, c)
        QMessageBox.information(
            self,
            "屏幕识别",
            "已保存，下次截屏时生效。",
        )
