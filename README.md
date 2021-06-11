
# 1. SPEECH EMOTION RECOGNITION
# 2. Giới thiệu đề tài
Nhận dạng cảm xúc kết hợp tiếng nói và văn bản có nghĩa là thông qua một tín hiệu tiếng nói đã biết và văn bản đại diện cho tiếng nói đó bằng các phương pháp chuyên môn, xử lý tín hiệu và văn bản, sau đó đưa ra kết luận về cảm xúc chứa đựng trong tín hiệu tiếng nói đó. Ví dụ như: vui, buồn, chán nản, xúc động, hạnh phúc … Nhận dạng cảm xúc tiếng nói có rất nhiều ứng dụng trong thực tế:
- Trong tương tác người – máy, robot có thể được dạy để có thể tương tác được với con người và nhận diện được cảm xúc của con người. một vật nuôi bằng robot có thể hiểu được không chỉ là những câu mệnh lệnh, mà còn cả những thông tin khác, như trạng thái tình cảm hay tình trạng sức khỏe chứa đựng trong câu mệnh lệnh đó để có những hành động tương ứng.
- Trong các tổng đài thông minh, nhận dạng cảm xúc tiếng nói giúp phát hiện những vấn đề tiềm tàng xuất hiện từ sự không hài lòng của khách hàng. 

<img src ='/display/home.png'>
Bạn chỉ cần đưa 1 file âm thanh và nhập vào chuỗi văn bản đại diện cho âm thanh đó, hệ thống sẽ tự động phân tích, nhận dạng cảm xúc chứa trong tín hiệu tiếng nói đó. Hiện tại mới nhận diện được 4 cảm xúc: vui, buồn, tức giận, bình thường với độ chính xác xấp xỉ 81%.

## 2.1. Mục đích:
Mặc dù cảm xúc là chủ quan và việc đánh nhãn cho dữ liệu rất khó khăn nhưng nhận dạng cảm xúc từ tín hiệu tiếng nói là một dự án đang được các nhà khoa học nghiên cứu, có tính ứng dụng và thực tiễn cao trong tương tác người máy- robot, ngành ngân hàng, dạy học,... Hiện nay các bộ dataset về speech emotion rất ít, và các bài báo khoa học mới nhất cũng mới chỉ nhận dạng được cảm xúc với độ chính xác dưới 80%. Tôi đã sử dụng 1 số thủ thuật và kiến trúc model mới nhất như : CNN kết hợp attention, BERT, augmentation để tăng độ chính xác lên xấp sỉ 81%.
## 2.2. Mục tiêu:
Thực nghiệm Speech Emotion Recognition trên các model truyền thống như : LSTM, LR, RF, MLP

Thực nghiệm Speech Emotion Recognition trên kiến trúc model: CNN + attention kết hợp với tăng cường dữ liệu (Augmentation)

Thực nghiệm Emotion Recognition trên chuỗi text sử dụng mô hình BERT

Xây dựng ứng dụng cơ bản: Nhận dạng cảm xúc kết hợp Tiếng nói và Văn bản

## 2.3. Thành viên
Trần Trung Trực -16T3
## 2.4. Dataset:
IEMOCAP dataset : https://sail.usc.edu/iemocap/
Thông tin chung:
- Ngôn ngữ: tiếng anh
- 10 diễn viên: 5 nam và 5 nữ
- 5 session với 5 cặp diễn viên
## 2.5. Thực nghiệm:
- Kết quả nhận diện cảm xúc từ tiếng nói khi chưa tăng cường dữ liệu (độ đo accuracy).
<img src ='/display/experiment_1.png'>

- Kết quả nhận diện cảm xúc từ tiếng nói khi tăng cường dữ liệu (độ đo accuracy).
<img src ='/display/experiment_2.png'>

- Kết quả nhận diện cảm xúc từ văn bản khi áp dụng các mô hình truyền thống.
<img src ='/display/experiment_3.png'>

- Kết quả nhận diện cảm xúc từ văn bản khi áp dụng mô hình BERT.
<img src ='/display/experiment_4.png'>

## 2.6. Get model:
Get model pretrain BERT at: https://drive.google.com/file/d/1nT49JK8SmJaxFJ5AF3ig4xA2YAfoXZHV/view?usp=sharing
Then save to '/static/models/'

# 3. Giao diện ứng dụng
** Giao diện home **

<img src ='/display/home.png'>
** Giao diện speech emotion **

<img src ='/display/speech.png'>
** Giao diện text emotion **

<img src ='/display/text.png'>
** Giao diện combine speech and text **

<img src ='/display/combine.png'>

# 4. Môi trường cài đặt 
- **python3 **
- tensorflow: pip install tensorflow
- pytorch: https://pytorch.org/get-started/locally/
- flask: pip install flask
- transformers: pip install transformers
- pip install -r requirements.txt
- Ngoài ra có thể có 1 số thư viện có thể cài thêm trong quá trình chạy
# 5. Hướng dẫn chạy
- create envs skin : conda create -n ser python==3.6
- activate skin :   source activate ser
-
- open directory project and open terminal: git clone https://github.com/trungtruc123/SER.git
- pip install -r requirements.txt
- python app.py

** get data test at: /static/data_test **
# 6. Hướng phát triển
- Thu thập, mở rộng thêm bộ dữ liệu
- Tăng độ chính xác của mô hình
- Phát triển thành một sản phẩm hoàn chỉnh
- Áp dụng nhận dạng cảm xúc cho tiếng việt
# 7. Tài liệu tham khảo
https://paperswithcode.com/paper/speech-emotion-recognition-with-multiscale

https://phamdinhkhanh.github.io/2020/05/23/BERTModel.html

https://paperswithcode.com/paper/multimodal-speech-emotion-recognition-using