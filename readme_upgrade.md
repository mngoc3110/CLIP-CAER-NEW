# Chi tiết các cải tiến so với baseline

Repo này được nâng cấp từ baseline [CLIP-CAER](https://github.com/zgsfer/CLIP-CAER) với các chiến lược huấn luyện nâng cao để cải thiện hiệu suất, bao gồm **Expression-Aware Adapters (EAA)**, **Instance-Enhanced Classifiers (IEC)**, **Mutual Information (MI) Loss**, **Decorrelation (DC) Loss**, và các phương pháp xử lý mất cân bằng dữ liệu.

## Tổng quan kiến trúc mới

Kiến trúc nâng cao được xây dựng dựa trên sườn của CLIP-CAER, kết hợp thêm nhiều module mới để tạo ra một mô hình mạnh mẽ và chính xác hơn.

### 1. Backbone Thị giác Hai luồng (Dual-Stream)
Mô hình xử lý hai luồng hình ảnh riêng biệt:
- **Luồng Gương mặt (Face Stream):** Các vùng mặt được cắt (crop) để nắm bắt các biểu cảm chi tiết, tinh vi.
- **Luồng Ngữ cảnh (Context Stream):** Toàn bộ khung hình (full-frame) được sử dụng để nắm bắt bối cảnh và hành vi xung quanh (có thể bật/tắt bằng cờ `--crop-body`).

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
- **Góc nhìn "Mô tả" thủ công (Hand-Crafted "Descriptive" View):** Các prompt mô tả chi tiết, giàu thông tin cho mỗi lớp cảm xúc, tập trung vào các biểu hiện vi mô trên gương mặt giống như Action Unit (AU) (ví dụ: "A person with furrowed eyebrows and a puzzled gaze"). Các prompt này là cố định.
- **Góc nhìn "Mềm" có thể học (Learnable "Soft" View):** Các vector ngữ cảnh theo kiểu CoOp có thể được tối ưu trong quá trình huấn luyện.
- **Mutual Information (MI) Loss:** Một hàm loss dựa trên InfoNCE được sử dụng để tối đa hóa thông tin tương hỗ (mutual information) giữa các embedding của prompt mô tả và prompt mềm (`t_desc` và `t_soft`), đảm bảo các prompt học được luôn bám sát ngữ nghĩa gốc.

### 5. IEC (Instance-Enhanced Classifier)
Để làm cho bộ phân loại dựa trên văn bản có khả năng thích ứng tốt hơn với các đặc trưng thị giác của từng mẫu video cụ thể, module IEC được sử dụng.
- **Cách triển khai:** Thay vì dùng một mẫu văn bản (text prototype) tĩnh cho mỗi lớp, một prototype động, được "tăng cường" theo từng mẫu, được tạo ra bằng cách sử dụng phép nội suy tuyến tính cầu (**Spherical Linear Interpolation - Slerp**).
- **Công thức:** `t_mix(k) = slerp(t_desc(k), z, λ_slerp)`, trong đó `t_desc(k)` là prompt mô tả cho lớp `k`, `z` là embedding thị giác của mẫu video, và `λ_slerp` là một trọng số có thể điều chỉnh.
- Việc phân loại cuối cùng được thực hiện bằng cách tính toán độ tương đồng giữa embedding thị giác `z` và các text prototype đã được trộn `t_mix` này.

### 6. Hàm Loss Tổng hợp và các Chiến lược Huấn luyện Nâng cao
Mô hình được huấn luyện với một hàm loss tổng hợp và nhiều kỹ thuật tiên tiến:
`L_total = L_classification + (weight_mi * L_mi) + (weight_dc * L_dc)`
- **`L_classification`**: Hàm loss cross-entropy tiêu chuẩn. Hỗ trợ nhiều cơ chế xử lý mất cân bằng dữ liệu:
    - **Class-Balanced Loss:** Tự động gán trọng số cao hơn cho các lớp thiểu số. Kích hoạt bằng cờ `--class-balanced-loss`.
    - **Logit Adjustment:** Điều chỉnh trực tiếp đầu ra logit của mô hình dựa trên tần suất xuất hiện của các lớp. Kích hoạt bằng cờ `--logit-adj`.
    - **WeightedRandomSampler:** Lấy mẫu các batch huấn luyện một cách có trọng số để đảm bảo các lớp thiểu số xuất hiện nhiều hơn. Kích hoạt bằng cờ `--use-weighted-sampler`.
    - **Label Smoothing:** Kỹ thuật regularize giúp giảm sự tự tin thái quá của mô hình. Kích hoạt bằng `--label-smoothing [0.0...1.0]`.
- **`L_mi` (Mutual Information Loss)**: Regularize quá trình học prompt. Trọng số được điều khiển bởi `--lambda_mi`.
- **`L_dc` (Decorrelation Loss)**: Khuyến khích các prompt của các lớp khác nhau trở nên khác biệt. Trọng số được điều khiển bởi `--lambda_dc`.
- **Loss Warmup & Ramp-up:** Trọng số của MI loss và DC loss được tăng dần trong quá trình huấn luyện để tăng tính ổn định, được điều khiển bởi các tham số `--mi-warmup`, `--mi-ramp`, `--dc-warmup`, `--dc-ramp`.
- **Automatic Mixed Precision (AMP):** Tăng tốc độ huấn luyện và giảm bộ nhớ GPU bằng cách sử dụng độ chính xác 16-bit. Kích hoạt bằng cờ `--use-amp`.
- **Gradient Clipping:** Giới hạn độ lớn của gradient để ngăn chặn hiện tượng "bùng nổ gradient" và ổn định quá trình huấn luyện. Kích hoạt bằng `--grad-clip [giá trị]`.

### 7. Huấn luyện theo giai đoạn (Staged Training)
Để đảm bảo quá trình huấn luyện ổn định hơn và hội tụ tốt hơn cho các mô hình phức tạp, kỹ thuật huấn luyện theo giai đoạn đã được triển khai. Quá trình này chia việc huấn luyện thành ba giai đoạn chính, mỗi giai đoạn tập trung vào việc huấn luyện một tập hợp các tham số khác nhau của mô hình:

-   **Giai đoạn 1: Huấn luyện Prompt Learner.** Chỉ huấn luyện các prompt có thể học (`learnable prompt`). Tất cả các phần khác của mô hình (bộ mã hóa ảnh, adapter, các lớp temporal) sẽ bị đóng băng. Mục tiêu là giúp các prompt text học cách liên kết với các đặc trưng thị giác cơ bản một cách ổn định.
-   **Giai đoạn 2: Huấn luyện Adapter và các Module Temporal.** Prompt learner đã được huấn luyện và bộ mã hóa ảnh được đóng băng, sau đó Adapter, Temporal Models và lớp Fusion được huấn luyện. Giai đoạn này giúp các module thị giác học cách trích xuất các đặc trưng không gian-thời gian quan trọng.
-   **Giai đoạn 3: Tinh chỉnh Toàn bộ (End-to-End Fine-tuning).** Tất cả các thành phần của mô hình được mở băng và huấn luyện cùng nhau với một learning rate rất nhỏ để tất cả các module có thể phối hợp với nhau một cách tốt nhất.

### 8. Logging chi tiết
Quá trình huấn luyện giờ đây sẽ in ra các thông tin chi tiết hơn sau mỗi epoch, bao gồm:
-   `Train WAR`, `Train UAR` của epoch hiện tại.
-   `Valid WAR`, `Valid UAR` của epoch hiện tại.
-   `Best Train WAR`, `Best Train UAR` tốt nhất từ đầu đến giờ.
-   `Best Valid WAR`, `Best Valid UAR` tốt nhất từ đầu đến giờ.
-   Ma trận nhầm lẫn (Confusion Matrix) của tập validation sau mỗi epoch.

## Hướng dẫn Sử dụng

Quá trình huấn luyện có thể được tùy chỉnh với các tham số dòng lệnh mới để điều khiển các tính năng nâng cao.

### Local
```bash
bash train.sh
```

### Google Colab
```bash
bash train_colab.sh
```

Bạn có thể chỉnh sửa các file `.sh` hoặc truyền trực tiếp tham số vào `main.py`. Các tham số quan trọng đã được thêm vào:
- `--staged-training`: (cờ) Bật chế độ huấn luyện theo giai đoạn.
- `--epochs-stage1`, `--epochs-stage2`, `--epochs-stage3` (int): Số epoch cho mỗi giai đoạn.
- `--lr-stage1`, `--lr-stage2`, `--lr-stage3` (float): Tốc độ học cho mỗi giai đoạn.
- `--lambda_mi` (float): Trọng số cho Mutual Information loss.
- `--lambda_dc` (float): Trọng số cho Decorrelation loss.
- `--mi-warmup`, `--mi-ramp`, `--dc-warmup`, `--dc-ramp` (int): Các tham số cho việc warmup và ramp-up loss.
- `--lr-adapter` (float): Tốc độ học (learning rate) cho Expression-aware Adapter.
- `--slerp-weight` (float): Hệ số nội suy cho Instance-enhanced Classifier. Đặt bằng `0` để tắt IEC.
- `--temperature` (float): Nhiệt độ (tau) cho lớp phân loại cuối cùng.
- `--class-balanced-loss`: (cờ) Bật để sử dụng loss được cân bằng theo lớp.
- `--logit-adj`: (cờ) Bật để sử dụng Logit Adjustment.
- `--logit-adj-tau` (float): Hệ số nhiệt độ cho Logit Adjustment.
- `--use-weighted-sampler`: (cờ) Bật để sử dụng `WeightedRandomSampler`.
- `--label-smoothing` (float): Hệ số làm mượt nhãn (label smoothing).
- `--use-amp`: (cờ) Bật để sử dụng Automatic Mixed Precision.
- `--grad-clip` (float): Giá trị giới hạn cho gradient clipping (ví dụ: 1.0).
- `--crop-body`: (cờ) Bật để cắt vùng body thay vì dùng toàn bộ khung hình.