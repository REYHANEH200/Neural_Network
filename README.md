import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.transform import resize


##----------------------------------------------------------- مرحله اول : تعریف توابع پایه ---------------------------------------------

# تابع تولید تصویر هر حرف فارسی
def generate_persian_letter(letter, size=16):
    img = np.zeros((size, size))
    
    if letter == 'الف':
        img[4:9, 5] = 1
        img[2, 5] = 1
        img[2,4] = 1
        img[2:4,3]=1
        img[2, 6] = 1
        img[1:3,7]=1
    elif letter == 'ب':
        img[1:2, 0] = 1  # عمودی چپ
        img[2, 0:7] = 1  # افقی پایین
        img[1:2, 6] = 1  # عمودی راست
        img[4,3]=1
    elif letter == 'پ':
        img[1:2, 0] = 1  # عمودی چپ
        img[2, 0:7] = 1  # افقی پایین
        img[1:2, 6] = 1  # عمودی راست
        img[4,2:5]=1
        img[6,3]=1
    elif letter == 'ت':
        img[2:3, 0] = 1  # عمودی چپ
        img[3, 0:7] = 1  # افقی پایین
        img[2:3, 6] = 1  # عمودی راست
        img[1 , 2:5]=1
    elif letter == 'ث':
        img[3:4, 0] = 1  # عمودی چپ
        img[4, 0:7] = 1  # افقی پایین
        img[3:4, 6] = 1  # عمودی راست
        img[2 , 2:5]=1
        img[0,3]=1
    elif letter == 'ج':
        img[3,2]=1
        img[2,3:5]=1
        img[3,5]=1
        img[4,4]=1
        img[5,3]=1
        img[6,2]=1
        img[7,2]=1
        img[9,4:7]=1
        img[8,3]=1
        img[8,7]=1
        img[6,5]=1
    elif letter == 'چ':
        img[3,2]=1
        img[2,3:5]=1
        img[3,5]=1
        img[4,4]=1
        img[5,3]=1
        img[6,2]=1
        img[7,2]=1
        img[9,4:8]=1
        img[8,3]=1
        img[8,8]=1
        img[6,5:8]=1
        img[7,6]=1
    elif letter == 'ح':
        img[3,2]=1
        img[2,3:5]=1
        img[3,5]=1
        img[4,4]=1
        img[5,3]=1
        img[6,2]=1
        img[7,2]=1
        img[9,4:7]=1
        img[8,3]=1
        img[8,7]=1
    elif letter == 'خ':
        img[0,3]=1
        img[3,2]=1
        img[2,3:5]=1
        img[3,5]=1
        img[4,4]=1
        img[5,3]=1
        img[6,2]=1
        img[7,2]=1
        img[9,4:7]=1
        img[8,3]=1
        img[8,7]=1
    elif letter == 'د':
        img[2,2:5]=1
        img[3,5]=1
        img[4,6]=1
        img[5:8 ,7]=1
        img[8,6]=1
        img[9,2:6]=1
    elif letter == 'ذ':
        img[0,3]=1
        img[2,2:5]=1
        img[3,5]=1
        img[4,6]=1
        img[5:8 ,7]=1
        img[8,6]=1
        img[9,2:6]=1
    elif letter == 'ر':
        img[1:5,7]=1
        img[5,7]=1
        img[6,6]=1
        img[7,5]=1
        img[8,3:5]=1
    
    return img

# plt.imshow(generate_persian_letter('الف'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('ب'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('پ'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('ت'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('ج'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('چ'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('خ'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('د'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('ذ'), cmap='gray')
# plt.show()
# plt.imshow(generate_persian_letter('ر'), cmap='gray')
# plt.show()

# تابع تولید داده‌های آموزشی
def generate_persian_data(num_samples=50, noise_level=0):
    X = []
    y = []
    base_size = 16  # اندازه ثابت خروجی
    
    for letter in persian_letters:
        base_img = generate_persian_letter(letter, size=base_size)
        
        for _ in range(num_samples):
            # اعمال چرخش
            angle = np.random.uniform(-15, 15)
            rotated = rotate(base_img, angle, reshape=False, mode='constant', cval=0)
            
            # تغییر اندازه با کنترل ابعاد
            scale = np.random.uniform(0.8, 1.2)
            new_size = int(base_size * scale)
            
            # محدود کردن اندازه‌ها برای جلوگیری از بزرگتر شدن از base_size
            new_size = min(new_size, base_size)
            
            # تغییر اندازه
            if new_size != base_size:
                scaled = resize(rotated, (new_size, new_size), mode='constant', anti_aliasing=True)
                
                # قرار دادن تصویر در مرکز
                final_img = np.zeros((base_size, base_size))
                h_start = (base_size - new_size) // 2
                w_start = (base_size - new_size) // 2
                final_img[h_start:h_start+new_size, w_start:w_start+new_size] = scaled
            else:
                final_img = rotated
            
            # اضافه کردن نویز
            noise_mask = np.random.random((base_size, base_size)) < noise_level
            noisy_img = np.where(noise_mask, 1 - final_img, final_img)
            
            X.append(noisy_img.flatten())
            y.append(letter)
    
    return np.array(X), np.array(y)
# ---------------------------------------- مرحله دوم : تعریف مدل‌های شبکه عصبی -------------------------------------------------------

class MultiClassPerceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros(num_classes)
    
    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        scores = np.dot(x, self.weights) + self.bias
        return np.argmax(scores, axis=1)
    
    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                true_class = np.argmax(y[i])
                
                scores = np.dot(x, self.weights) + self.bias
                pred_class = np.argmax(scores)
                
                if pred_class != true_class:
                    self.weights[:, true_class] += learning_rate * x
                    self.bias[true_class] += learning_rate
                    self.weights[:, pred_class] -= learning_rate * x
                    self.bias[pred_class] -= learning_rate

class DeltaRuleNetwork:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros(num_classes)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.argmax(self.softmax(np.dot(x, self.weights) + self.bias), axis=1)
    
    def train(self, X, y, learning_rate=0.1, epochs=100):
        errors = []
        for epoch in range(epochs):
            scores = np.dot(X, self.weights) + self.bias
            probs = self.softmax(scores)
            
            error = y - probs
            grad_weights = np.dot(X.T, error)
            grad_bias = np.sum(error, axis=0)
            
            self.weights += learning_rate * grad_weights
            self.bias += learning_rate * grad_bias
            
            errors.append(np.mean(np.sum(error**2, axis=1)))
        
        return errors

#------------------------------------------ مرحله سوم :  تولید و آماده‌سازی داده‌ها --------------------------------------------------------

# تعریف حروف فارسی مورد استفاده
persian_letters = ['الف', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر']

# تولید داده‌های تمیز
X_clean, y_clean = generate_persian_data(noise_level=0)

# تولید داده‌های نویزی با سطوح مختلف
X_noisy10, y_noisy10 = generate_persian_data(noise_level=0.1)
X_noisy20, y_noisy20 = generate_persian_data(noise_level=0.2)
X_noisy50, y_noisy50 = generate_persian_data(noise_level=0.5)

# تبدیل برچسب‌ها به فرم عددی
lb = LabelBinarizer()
y_clean_encoded = lb.fit_transform(y_clean)

# تقسیم داده به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean_encoded, test_size=0.2)

# ------------------------------------------- مرحله چهارم : آموزش مدل‌ها -----------------------------------------------------------------

input_size = X_train.shape[1]
num_classes = y_train.shape[1]

# ایجاد مدل‌ها
perceptron = MultiClassPerceptron(input_size, num_classes)
delta_rule = DeltaRuleNetwork(input_size, num_classes)

# آموزش پرسپترون
perceptron.train(X_train, y_train, epochs=200)

# آموزش قانون دلتا و ذخیره خطاها
delta_errors = delta_rule.train(X_train, y_train, epochs=200)

# -------------------------------------------- مرحله پنجم : ارزیابی مدل‌ها ---------------------------------------------------------------

def evaluate(model, X, y, model_type='perceptron'):
    if model_type == 'perceptron':
        preds = model.predict(X)
    else:
        preds = model.predict(X)
    
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(preds == true_labels) * 100
    return accuracy

# محاسبه دقت‌ها
results = {
    'Perceptron': {
        'clean': evaluate(perceptron, X_test, y_test),
        'noisy10': evaluate(perceptron, X_noisy10, lb.transform(y_noisy10)),
        'noisy20': evaluate(perceptron, X_noisy20, lb.transform(y_noisy20)),
        'noisy50': evaluate(perceptron, X_noisy50, lb.transform(y_noisy50))
    },
    'DeltaRule': {
        'clean': evaluate(delta_rule, X_test, y_test, 'delta'),
        'noisy10': evaluate(delta_rule, X_noisy10, lb.transform(y_noisy10), 'delta'),
        'noisy20': evaluate(delta_rule, X_noisy20, lb.transform(y_noisy20), 'delta'),
        'noisy50': evaluate(delta_rule, X_noisy50, lb.transform(y_noisy50), 'delta')
    }
}

# ------------------------------------------------------- مرحله ششم : نمایش نتایج --------------------------------------------------------------

print("\nEvaluation Results:")
print("-------------------------------------")
print(f"Perceptron Accuracy on Clean Data: {results['Perceptron']['clean']:.2f}%")
print(f"Delta Rule Accuracy on Clean Data: {results['DeltaRule']['clean']:.2f}%")
print("-------------------------------------")
print(f"Perceptron Accuracy on Noisy Data 10%: {results['Perceptron']['noisy10']:.2f}%")
print(f"Delta Rule Accuracy on Noisy Data 10%: {results['DeltaRule']['noisy10']:.2f}%")
print("-------------------------------------")
print(f"Perceptron Accuracy on Noisy Data 20%: {results['Perceptron']['noisy20']:.2f}%")
print(f"Delta Rule Accuracy on Noisy Data 20%: {results['DeltaRule']['noisy20']:.2f}%")
print("-------------------------------------")
print(f"Perceptron Accuracy on Noisy Data 50%: {results['Perceptron']['noisy50']:.2f}%")
print(f"Delta Rule Accuracy on Noisy Data 50%: {results['DeltaRule']['noisy50']:.2f}%")
print("-------------------------------------")

# رسم نمودار خطای قانون دلتا
plt.figure(figsize=(10, 6))
plt.plot(delta_errors)
plt.title('Error Reduction Process in the Delta Rule')
plt.xlabel('Training Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# نمایش نمونه‌ای از پیش‌بینی‌ها
print("\nSample Predictions:")
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    true_label = persian_letters[np.argmax(y_test[idx])]
    p_pred = persian_letters[perceptron.predict(X_test[idx])[0]]
    d_pred = persian_letters[delta_rule.predict(X_test[idx])[0]]
    print(f"True Output: {true_label} | Perceptron: {p_pred} | Delta Rule: {d_pred}")
