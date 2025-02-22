import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    return data

# 训练模型并可视化
def train_model(data, previous_model=None):
    X = [item['features'] for item in data]  # 假设数据中有'features'字段
    y = [item['label'] for item in data]      # 假设数据中有'label'字段
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = previous_model if previous_model else RandomForestClassifier()  # 使用之前的模型
    
    # 记录训练过程中的准确率
    train_accuracies = []
    test_accuracies = []
    
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='Train Accuracy', color='blue')
    line2, = ax.plot([], [], label='Test Accuracy', color='orange')
    ax.set_xlim(1, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Training Accuracy')
    ax.legend()
    
    for i in range(1, 301):  # 训练300次
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # 更新图表
        line1.set_xdata(range(1, i + 1))
        line1.set_ydata(train_accuracies)
        line2.set_xdata(range(1, i + 1))
        line2.set_ydata(test_accuracies)
        
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)  # 暂停以更新图表
        
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图表
    
    return model

# 反馈机制
def feedback(model, new_data):
    # 假设new_data是一个包含新特征和标签的列表
    X_new = [item['features'] for item in new_data]
    y_new = [item['label'] for item in new_data]
    model.fit(X_new, y_new)  # 增量学习

# 主程序
if __name__ == "__main__":
    data = load_data('./data/data.json')
    model = train_model(data)  # 初始训练
    
    # 示例反馈数据（可以根据需要手动添加）
    new_feedback_data = [
        {"features": [20.0, 10.1], "label": 0},
        {"features": [32.0, 18.65], "label": 1},
        {"features": [23.25, 9.7], "label": 0},
        {"features": [35.55, 20.25], "label": 1},
        {"features": [18.7, 16.85], "label": 0},
        {"features": [27.3, 5.5], "label": 1}
    ]
    
    feedback(model, new_feedback_data)  # 增量学习
    model = train_model(data, model)  # 使用之前的模型进行增量学习