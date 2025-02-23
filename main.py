# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    return data

# 训练模型并可视化
def train_model(data, previous_model=None):
    X = [item['features'] for item in data]
    y = [item['label'] for item in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 调试信息
    print(f"Length of X_train: {len(X_train)}, Length of y_train: {len(y_train)}")
    
    model = previous_model if previous_model else RandomForestClassifier()
    
    # 记录训练过程中的准确率和损失率
    train_accuracies = []
    test_accuracies = []
    losses = []
    
    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='Train Accuracy', color='blue')
    line2, = ax.plot([], [], label='Test Accuracy', color='orange')
    line3, = ax.plot([], [], label='Loss', color='red')
    #ax.set_xlim(0, 500)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Accuracy / Loss')
    ax.set_title('Model Training Accuracy and Loss')
    ax.legend()
    

    
    for i in range(1, 501):
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        loss = 1 - test_accuracy
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        losses.append(loss)
        
        # 更新图表
        line1.set_xdata(range(1, i + 1))
        line1.set_ydata(train_accuracies)
        line2.set_xdata(range(1, i + 1))
        line2.set_ydata(test_accuracies)
        line3.set_xdata(range(1, i + 1))
        line3.set_ydata(losses)
        
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)  # 暂停以更新图表
        
    plt.ioff()
    plt.show()
    

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
        {"features": [12.3, 24.2], "label": 0},
        {"features": [21.1, 15.4], "label": 1},
        {"features": [8.12, 22.45], "label": 0},
        {"features": [21.5, 20.25], "label": 1},
        {"features": [15.2, 12.3], "label": 0},
        {"features": [18.1, 19.2], "label": 1},
        {"features": [10.5, 12.7], "label": 0},
        {"features": [20.2, 22.1], "label": 1},
        {"features": [14.3, 16.4], "label": 0},
        {"features": [17.5, 18.2], "label": 1}
    ]

    while True:
        feedback(model, new_feedback_data)  # 增量学习
        model = train_model(data, model)  # 使用之前的模型进行增量学习