# Chi tiết các cải tiến so với baseline

Repo này được nâng cấp từ baseline [CLIP-CAER](https://github.com/zgsfer/CLIP-CAER) với các chiến lược huấn luyện nâng cao để cải thiện hiệu suất, bao gồm **Expression-Aware Adapters (EAA)**, **Instance-Enhanced Classifiers (IEC)**, **Mutual Information (MI) Loss**, **Decorrelation (DC) Loss**, và các phương pháp xử lý mất cân bằng dữ liệu.

## Tổng quan kiến trúc mới

Kiến trúc nâng cao được xây dựng dựa trên sườn của CLIP-CAER, kết hợp thêm nhiều module mới để tạo ra một mô hình mạnh mẽ và chính xác hơn.

### 1. Backbone Thị giác Hai luồng (Dual-Stream)
Mô hình xử lý hai luồng hình ảnh riêng biệt:
- **Luồng Gương mặt (Face Stream):** Các vùng mặt được cắt (crop) để nắm bắt các biểu cảm chi tiết, tinh vi.
- **Luồng Ngữ cảnh (Context Stream):** Toàn bộ khung hình hoặc vùng cơ thể để nắm bắt bối cảnh và hành vi xung quanh.

Cả hai luồng đều được xử lý bởi cùng một bộ mã hóa hình ảnh của CLIP (CLIP Visual Encoder).

### 2. EAA (Expression-Aware Adapter)
Để nắm bắt tốt hơn các chi tiết cảm xúc tinh vi trên gương mặt mà không làm mất đi khả năng tổng quát hóa của mô hình CLIP đã được huấn luyện trước, một module **Expression-Aware Adapter** gọn nhẹ được tích hợp vào luồng xử lý gương mặt.
- **Cách triển khai:** Một module adapter theo kiến trúc "bottleneck" được chèn vào sau bộ mã hóa hình ảnh của luồng gương mặt.
- **Trainable:** Chỉ các tham số của adapter được tinh chỉnh (fine-tune), trong khi phần lớn bộ mã hóa hình ảnh được giữ đông lạnh (frozen).

### 3. Mô hình hóa Thời gian và Kết hợp (Temporal Modeling and Fusion)
- **Bộ mã hóa thời gian (Temporal Encoders):** Chuỗi các đặc trưng (features) từ mỗi khung hình của cả hai luồng được đưa qua các mô hình Temporal Transformer riêng biệt để nắm bắt mối quan hệ theo thời gian.
- **Kết hợp (Fusion):** Các embedding cấp độ video cho luồng mặt (`z_f`) và luồng ngữ cảnh (`z_c`) sau đó được kết hợp bằng cách ghép nối (concatenation) và qua một lớp chiếu (linear projection) để tạo ra một embedding thị giác cuối cùng, hợp nhất là `z`.

### 4. Prompt Hai góc nhìn (Dual-View Prompting) & MI Loss
Để ngăn các prompt có thể học (learnable prompts) bị overfitting và đi chệch khỏi ngữ nghĩa mong muốn, một chiến lược prompt hai góc nhìn được sử dụng.
- **Góc nhìn "Mô tả" thủ công (Hand-Crafted "Descriptive" View):** Các prompt mô tả chi tiết, giàu thông tin cho mỗi lớp cảm xúc, mô tả cả hành vi và các biểu hiện vi mô trên gương mặt giống như Action Unit (AU) (ví dụ: "Một người với cặp lông mày nhíu lại và ánh mắt bối rối"). Các prompt này là cố định.
- **Góc nhìn "Mềm" có thể học (Learnable "Soft" View):** Các vector ngữ cảnh theo kiểu CoOp có thể được tối ưu trong quá trình huấn luyện.
- **Mutual Information (MI) Loss:** Một hàm loss dựa trên InfoNCE được sử dụng để tối đa hóa thông tin tương hỗ (mutual information) giữa các embedding của prompt mô tả và prompt mềm (`t_desc` và `t_soft`), đảm bảo các prompt học được luôn bám sát ngữ nghĩa gốc.

### 5. IEC (Instance-Enhanced Classifier)
Để làm cho bộ phân loại dựa trên văn bản có khả năng thích ứng tốt hơn với các đặc trưng thị giác của từng mẫu video cụ thể, module IEC được sử dụng.
- **Cách triển khai:** Thay vì dùng một mẫu văn bản (text prototype) tĩnh cho mỗi lớp, một prototype động, được "tăng cường" theo từng mẫu, được tạo ra bằng cách sử dụng phép nội suy tuyến tính cầu (**Spherical Linear Interpolation - Slerp**).
- **Công thức:** `t_mix(k) = slerp(t_desc(k), z, λ_slerp)`, trong đó `t_desc(k)` là prompt mô tả cho lớp `k`, `z` là embedding thị giác của mẫu video, và `λ_slerp` là một trọng số có thể điều chỉnh.
- Việc phân loại cuối cùng được thực hiện bằng cách tính toán độ tương đồng giữa embedding thị giác `z` và các text prototype đã được trộn `t_mix` này.

### 6. Hàm Loss Tổng hợp (Composite Loss Function)
Mô hình được huấn luyện với một hàm loss tổng hợp:
`L_total = L_classification + λ_mi * L_mi + λ_dc * L_dc`
- **`L_classification`**: Hàm loss cross-entropy tiêu chuẩn cho tác vụ phân loại chính. Có hai tùy chọn để xử lý mất cân bằng dữ liệu:
    - **Class-Balanced Loss:** Tự động gán trọng số cao hơn cho các lớp thiểu số (ít mẫu).
    - **Logit Adjustment:** Điều chỉnh trực tiếp đầu ra logit của mô hình dựa trên tần suất xuất hiện của các lớp.
- **`L_mi` (Mutual Information Loss)**: Regularize quá trình học prompt.
- **`L_dc` (Decorrelation Loss)**: Khuyến khích các prompt của các lớp khác nhau trở nên khác biệt, giảm sự tương quan và tránh việc các embedding bị "sụp đổ" vào một điểm.

## Hướng dẫn Sử dụng

Quá trình huấn luyện có thể được tùy chỉnh với các tham số dòng lệnh mới để điều khiển các tính năng nâng cao.

```bash
bash train.sh
```
Bạn có thể chỉnh sửa file `train.sh` hoặc truyền trực tiếp tham số vào `main.py`. Các tham số mới bao gồm:
- `--mi-loss-weight` (float, mặc định: 0.5): Trọng số cho Mutual Information loss.
- `--dc-loss-weight` (float, mặc định: 0.1): Trọng số cho Decorrelation loss.
- `--lr-adapter` (float, mặc định: 1e-4): Tốc độ học (learning rate) cho Expression-aware Adapter.
- `--slerp-weight` (float, mặc định: 0.5): Hệ số nội suy cho Instance-enhanced Classifier. Đặt bằng `0` để tắt IEC.
- `--temperature` (float, mặc định: 0.07): Nhiệt độ (tau) cho lớp phân loại cuối cùng.
- `--class-balanced-loss`: (cờ) Bật để sử dụng loss được cân bằng theo lớp.
- `--logit-adj`: (cờ) Bật để sử dụng Logit Adjustment.
- `--logit-adj-tau` (float, mặc định: 1.0): Hệ số nhiệt độ cho Logit Adjustment.
