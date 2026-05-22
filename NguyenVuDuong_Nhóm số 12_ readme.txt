I. THÔNG TIN THÀNH VIÊN NHÓM, PHÂN CÔNG CÔNG VIỆC VÀ KỊCH BẢN THỰC NGHIỆM.

1. THÔNG TIN THÀNH VIÊN NHÓM
- Thành viên 1: Nguyễn Vũ Dương (nhóm trưởng)| MSSV: 23001964 | Công việc: tiền xử lý, nội dung và mô hình SVM, từ SVM sang SVR (hồi quy), file Báo cáo, file mã nguồn, slide.
- Thành viên 2: Nguyễn Mạnh Giang 	     | MSSV: 23000113 | Công việc: nội dung và mô hình SoftMax, KNN , từ SoftMax phân loại sang hồi quy, slide.
- Thành viên 3: Bùi Tiến Bình     	     | MSSV: 23000095 | Công việc: nội dung và mô hình phân cụm Kmeans, giảm chiều PCA/LDA, slide.


2. Cách tổ chức chương trình

Chương trình được tổ chức gồm:
01 file notebook Jupyter Notebook (.ipynb) chứa toàn bộ mã nguồn thực nghiệm, chứa các phần như:
	- Đọc dữ liệu gốc đầu vào
	- tiền xử lý dữ liệu
	- giảm chiều và trực quan hóa
	- mô hình phân cụm dữ liệu
	- Các mô hình phân loại
	- Chuyển mô hình phân loại sang hồi quy
	- phân loại và đánh giá kết quả
Để chạy được chương trình thì ta phải cho tệp dữ liệu gốc vào phần đầu tiên sau đó chạy lần lượt từng phần.

3. KỊCH BẢN THỰC NGHIỆM

Nhằm đánh giá hiệu quả của các phương pháp học máy trên bộ dữ liệu sau tiền xử lý, nhóm tiến hành xây dựng các kịch bản thực nghiệm với nhiều mô hình khác nhau. Mỗi mô hình đều được huấn luyện và đánh giá trên ba dạng dữ liệu gồm: dữ liệu đã chuẩn hóa ban đầu, dữ liệu giảm chiều bằng PCA và dữ liệu giảm chiều bằng LDA. Các chỉ số đánh giá bao gồm Accuracy, F1-score và Confusion Matrix nhằm phân tích hiệu quả phân loại của từng phương pháp.

Kịch bản 1: Mô hình Softmax Regression

Tiến hành huấn luyện mô hình Softmax Regression trên các tập dữ liệu sau:
- Dữ liệu đã được chuẩn hóa (dữ liệu gốc).
- Dữ liệu sau khi giảm chiều bằng phương pháp PCA.
- Dữ liệu sau khi giảm chiều bằng phương pháp LDA.

Sau quá trình huấn luyện, mô hình được đánh giá thông qua các chỉ số:
- Accuracy: đánh giá độ chính xác tổng thể của mô hình.
- F1-score: đánh giá sự cân bằng giữa Precision và Recall.
- Confusion Matrix: phân tích khả năng phân loại đúng và sai giữa các lớp.

Kịch bản 2: Mô hình SVM (RBF Kernel)

Tiến hành huấn luyện mô hình Support Vector Machine (SVM) với RBF Kernel trên các tập dữ liệu:
- Dữ liệu đã được chuẩn hóa (dữ liệu gốc).
- Dữ liệu sau khi giảm chiều bằng PCA.
- Dữ liệu sau khi giảm chiều bằng LDA.

Hiệu quả của mô hình được đánh giá dựa trên:
- Accuracy.
- F1-score.
- Confusion Matrix.

Thông qua đó, nhóm đánh giá khả năng phân tách dữ liệu phi tuyến của SVM khi áp dụng trên các không gian đặc trưng khác nhau.

Kịch bản 3: Mô hình K-Nearest Neighbors (KNN)

Tiến hành huấn luyện mô hình K-Nearest Neighbors (KNN) với số lượng láng giềng gần nhất K = 7 trên các tập dữ liệu:
- Dữ liệu đã được chuẩn hóa (dữ liệu gốc).
- Dữ liệu sau khi giảm chiều bằng PCA.
- Dữ liệu sau khi giảm chiều bằng LDA.

Kết quả mô hình được đánh giá thông qua:
- Accuracy.
- F1-score.
- Confusion Matrix.

Từ đó, nhóm phân tích ảnh hưởng của việc giảm chiều dữ liệu đến hiệu quả phân loại của thuật toán KNN.

So sánh kết quả thực nghiệm

Sau khi hoàn thành các kịch bản trên, nhóm tiến hành so sánh hiệu quả giữa các mô hình dựa trên các chỉ số đánh giá và khả năng tổng quát hóa dữ liệu. Đồng thời, nhóm phân tích tác động của các phương pháp giảm chiều (PCA, LDA) đến hiệu suất của từng mô hình học máy.


II. link dữ liệu dự án.

Link bộ dữ liệu gốc: https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
Hoặc Thầy/Cô có thể tải trực tiếp từ link sau: https://drive.google.com/file/d/1lx8Or3Bdz1Fa6qkRymet2lXWCgA-d2lp/view?usp=drive_link

III. hướng dẫn cách tổ chức thư mục để thực nghiệm

Nhóm sẽ cung cấp cho thầy/cô một file mã nguồn định dạng .ipynb. Để thuận tiện trong quá trình chạy chương trình và tái hiện kết quả, nhóm khuyến nghị thầy/cô tải file này lên Google Colab. Sau đó, tải bộ dữ liệu gốc từ Kaggle theo đường link được nhóm cung cấp tại Phần I lên Google Colab và tiến hành chạy lần lượt từng phần của notebook theo thứ tự. Để rõ ràng hơn thầy/cô làm theo các bước sau:
	+) B1: Tải file mã nguồn lên Google Colab.
	+) B2: tải tệp dữ liệu gốc lên Google Colab.
	+) B3: chạy lần lượt từng phần trong file mã nguồn.












