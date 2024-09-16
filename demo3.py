from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# Đọc dữ liệu lương và số năm kinh nghiệm
salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
X = salary[['Experience Years']].values
y = salary['Salary'].values.reshape(-1, 1)

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu cho Neural Network
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)  # Chuẩn hóa y cho Neural Network

# Neural Network (MLPRegressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=5000, random_state=2529)
mlp_model.fit(X_train_scaled, y_train_scaled.ravel())  # Huấn luyện mô hình MLP

# Lasso Regression
lasso_model = Lasso(alpha=1.0, max_iter=1000, random_state=2529)  # Khởi tạo mô hình Lasso
lasso_model.fit(X_train, y_train.ravel())  # Huấn luyện mô hình Lasso trực tiếp trên dữ liệu thô, không chuẩn hóa


#Linear Resression

# X= X_train
# y = y_train


#tạo một ma trận có X.shape[0] hàng và 1 cột có toàn giá trị 1 ̣(hàng dọc)
# one = np.ones((X.shape[0], 1))
# Xbar = np.concatenate((one, X), axis = 1)


# Thêm cho tập huấn luyện
one_train = np.ones((X_train.shape[0], 1))
Xbar_train = np.concatenate((one_train, X_train), axis=1)

# # Thêm tập kiểm tra
# one_test = np.ones((X_test.shape[0], 1))
# Xbar_test = np.concatenate((one_test, X_test), axis=1)

#
A = np.dot(Xbar_train.T, Xbar_train)
b = np.dot(Xbar_train.T, y_train)
w = np.dot(np.linalg.pinv(A),b)


w_0=w[0]
w_1=w[1]




# Route chính để nhập số năm kinh nghiệm và hiển thị kết quả từ cả 2 mô hình
@app.route('/', methods=["POST", "GET"])
def hello_world():
    result = ''
    if request.method == "POST":
        input1 = request.form.get("name")
        if input1:
            try:
                # Chuyển đổi giá trị input thành mảng
                input_value = float(input1)
                final_features = np.array([[input_value]])

                # Dự đoán mức lương từ Neural Network
                final_features_scaled = scaler_X.transform(final_features)
                prediction_scaled_mlp = mlp_model.predict(final_features_scaled)
                prediction_mlp = scaler_y.inverse_transform(prediction_scaled_mlp.reshape(-1, 1))
                output_mlp = prediction_mlp[0][0]

                # Dự đoán mức lương từ Lasso (không chuẩn hóa)
                prediction_lasso = lasso_model.predict(final_features)  # Sử dụng dữ liệu thô, không chuẩn hóa
                output_lasso = prediction_lasso[0]
                output_linear_regression =  w_0 + w_1*input_value

                # Định dạng kết quả từ cả 2 mô hình
                result = f'Lương dự đoán (Neural Network) cho {input_value} năm kinh nghiệm: {output_mlp}<br>' \
                         f'Lương dự đoán (Lasso Regression) cho {input_value} năm kinh nghiệm: {output_lasso}<br>'\
                         f'Lương dự đoán (Linear Regression) cho {input_value} năm kinh nghiệm: {output_linear_regression}'
            except ValueError:
                result = "Vui lòng nhập một số hợp lệ!"
    return '''
            <html>
                <body>
                    <h2>Nhập số năm kinh nghiệm để dự đoán lương</h2>
                    <form method="POST" action="/">
                        <label for="name">Số năm kinh nghiệm:</label>
                        <input type="text" name="name">
                        <input type="submit" value="Dự đoán">
                    </form>
                    <h3>''' + result + '''</h3>
                </body>
            </html>
        '''

if __name__ == "__main__":
    app.run(debug=True)