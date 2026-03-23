


























#Sau khi giảm chiều, có 4 nhóm đánh giá chính:

#Nhóm 1 — Đánh giá chất lượng giảm chiều (không cần model)

#Explained Variance Ratio — PCA giữ lại bao nhiêu % thông tin. Thường yêu cầu ≥ 85%
#Scree Plot — đồ thị elbow để chọn số chiều tối ưu
#LDA Separation — trực quan hóa các class có tách biệt rõ sau LDA không (scatter plot LD1 vs LD2)


#Nhóm 2 — Đánh giá hiệu năng phân loại (quan trọng nhất)

#Accuracy, Precision, Recall, F1-score — dùng classification_report của sklearn
#Confusion Matrix — xem mô hình nhầm class nào
#Cross-validation — đánh giá tổng quát hóa (đã làm ở trên)


#Nhóm 3 — So sánh trước và sau giảm chiều

#So sánh accuracy baseline vs PCA vs LDA trực tiếp — đây là test thực tế nhất cho báo cáo


#Nhóm 4 — Kiểm định thống kê (nâng cao)

#t-test / Wilcoxon — kiểm tra xem sự chênh lệch accuracy giữa PCA và LDA có ý nghĩa thống kê không