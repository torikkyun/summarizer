from transformers import BartForConditionalGeneration, BartTokenizer

# Load lại mô hình và tokenizer đã fine-tune
model_name_or_path = "./bart_finetuned"
tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
model = BartForConditionalGeneration.from_pretrained(model_name_or_path)


def summarize(text, max_length=512, min_length=30):
    inputs = tokenizer(
        [text], max_length=512, truncation=True, return_tensors="pt"
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    text = 'Anh nhập Bệnh viện Đa khoa Xuyên Á TP HCM trong vòng 20 phút từ khi xuất hiện triệu chứng. Các bác sĩ ghi nhận bệnh nhân có dấu hiệu điển hình của đột quỵ, bao gồm chóng mặt, run nửa người bên phải, kèm theo huyết áp cao bất thường ở mức 200/100 mmHg. Kết quả chụp MRI xác định nhồi máu tiểu não bên phải. Ngay lập tức, bệnh nhân được kiểm soát huyết áp và tiêm thuốc tiêu sợi huyết (rTPA) qua đường tĩnh mạch. Loại thuốc này giúp tái thông mạch máu bị tắc, đảm bảo cung cấp máu và oxy kịp thời đến các tế bào não, ngăn chặn tổn thương nghiêm trọng. Ngay sau khi tiêm thuốc, tình trạng chóng mặt của bệnh nhân giảm rõ rệt, triệu chứng run cải thiện, có thể giữ thăng bằng khi ngồi và đi lại. Kết quả chụp CT mạch máu não không ghi nhận hẹp hay tắc mạch máu lớn, cho thấy hiệu quả của việc cấp cứu đúng thời điểm. Hiện, anh có thể tự đi lại và vận động bình thường, xuất viện sau 7 ngày điều trị. Bác sĩ Lê Minh Mẫn, Khoa Nội thần kinh, nhấn mạnh tầm quan trọng của việc nhận biết sớm các dấu hiệu đột quỵ để can thiệp kịp thời. Theo quy tắc F.A.S.T, các dấu hiệu điển hình bao gồm méo miệng, liệt tay, nói khó và cần xác định thời gian để đưa bệnh nhân đến bệnh viện sớm nhất. Tuy nhiên, trong một số trường hợp hiếm gặp như bệnh nhân trên, đột quỵ có thể biểu hiện bằng các triệu chứng không điển hình như chóng mặt, choáng váng, mất thăng bằng, mất thị lực, nhìn mờ hoặc nhìn đôi. Vì vậy, quy tắc nhận biết đột quỵ mở rộng là B.E.F.A.S.T (Balance - mất thăng bằng, Eyes - vấn đề thị lực, Face - méo miệng, Arm - liệt tay, Speech - nói khó, Time - thời gian). Bác sĩ Mẫn khuyến cáo khi phát hiện người thân hoặc người xung quanh có dấu hiệu đột quỵ, cần nhanh chóng đưa họ đến cơ sở y tế có trung tâm đột quỵ gần nhất. Tuyệt đối không áp dụng các phương pháp dân gian, vì điều này có thể làm mất "thời gian vàng", dẫn đến hậu quả nghiêm trọng như hôn mê hoặc thậm chí tử vong. Mỹ Ý'
    print("Tóm tắt:", summarize(text))
