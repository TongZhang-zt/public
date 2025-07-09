# 日志系统
import datetime
from collections import deque
import ipywidgets as widgets
from IPython.display import display, HTML

class LogManager:
    def __init__(self, max_logs=100):
        self.logs = deque(maxlen=max_logs)
        self.output_widget = widgets.Output(
            layout=widgets.Layout(
                width='260px',
                height='200px',
                border='1px solid gray',
                overflow='auto'
            )
        )
        
        # 清空日志按钮
        self.clear_button = widgets.Button(
            description='清空日志',
            button_style='warning',
            layout=widgets.Layout(width='100px', height='30px')
        )
        self.clear_button.on_click(self.clear_logs)
        
        # 导出日志按钮
        self.export_button = widgets.Button(
            description='导出日志',
            button_style='info',
            layout=widgets.Layout(width='100px', height='30px')
        )
        self.export_button.on_click(self.export_logs)
        
        # 日志控制面板
        self.control_panel = widgets.HBox([
            self.clear_button,
            self.export_button
        ])
        
        # 完整的日志组件
        self.log_widget = widgets.VBox([
            widgets.Label("运行日志:"),
            self.output_widget,
            self.control_panel
        ])
    
    def add_log(self, message, level="INFO"):
        """添加日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"  # 重新添加这一行
        self.logs.append(log_entry)
        
        with self.output_widget:
            if level == "ERROR":
                display(HTML(f'<p style="color:red; margin:0; font-family:monospace;">{log_entry}</p>'))
            elif level == "WARN":
                display(HTML(f'<p style="color:orange; margin:0; font-family:monospace;">{log_entry}</p>'))
            elif level == "SUCCESS":
                display(HTML(f'<p style="color:green; margin:0; font-family:monospace;">{log_entry}</p>'))
            else:
                display(HTML(f'<p style="margin:0; font-family:monospace;">{log_entry}</p>'))
    
    def clear_logs(self, btn=None):
        """清空日志"""
        self.logs.clear()
        self.output_widget.clear_output()
        self.add_log("日志已清空", "INFO")
    
    def export_logs(self, btn=None):
        """导出日志到文件"""
        try:
            filename = f"grap_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                for log in self.logs:
                    f.write(log + '\n')
            self.add_log(f"日志已导出到: {filename}", "SUCCESS")
        except Exception as e:
            self.add_log(f"导出日志失败: {str(e)}", "ERROR")

# 创建日志管理器
log_manager = LogManager()
log_manager.add_log("日志系统已初始化", "SUCCESS")