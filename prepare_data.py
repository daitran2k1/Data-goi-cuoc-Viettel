import re
import json
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


# lấy thông tin ban đầu
# with open('viettel.json', 'w') as f:
#     instruction = "Nếu bạn là nhân viên tổng đài của Viettel, hãy trả lời những câu hỏi về gói cước dựa trên mô tả của khách hàng."
#     input = "Tôi muốn biết thêm thông tin về gói cước {}".format()
#     output = "Thông tin của gói cước {} như sau: {}".format()
#     f.write("{\n")
#     f.write("   \"instruction\": \"{}\",\n".format(instruction))
#     f.write("   \"input\": \"{}\",\n".format(input))
#     f.write("   \"output\": \"{}\"\n".format(output))
#     f.write("},\n")
# new_lines = []
# with open('viettel_giagoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         pos = line.find('đ')
#         if pos == -1:
#             new_lines.append(line)
#             continue
#         new_lines.append(line[:pos+1]+'\n')
# with open('viettel_giagoicuoc1.txt', 'w', encoding='utf-8') as f:
#     f.writelines(new_lines)
# new_lines = []
# with open('viettel_dangkygoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         if 'gui' in line or 'gửi' in line:
#             if 'Cú pháp' in line:
#                 pos = line.find(':')
#                 goicuoc = str(line[:pos]).strip()
#                 thongtin = str(line[pos+1:]).strip()
#                 pos_ = thongtin.find('Cú pháp')
#                 thongtin = str(thongtin[pos_:])
#                 line = goicuoc + ': ' + thongtin + '\n'
#                 new_lines.append(line)
#                 continue
#             new_lines.append(line)
# with open('viettel_dangkygoicuoc1.txt', 'w', encoding='utf-8') as f:
#     f.writelines(new_lines)
# new_lines = []
# with open('viettel_dangkygoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         if 'hủy' in line or 'Hủy' in line:
#             new_lines.append(line)
# with open('viettel_huygiahangoicuoc.txt', 'w', encoding='utf-8') as f:
#     f.writelines(new_lines)
# with open('viettel_huygoicuoc.txt', 'w', encoding='utf-8') as f:
#     f.writelines(new_lines)


# tạo data
# with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     with open('viettel.json', 'w', encoding='utf-8') as f_data:
#         for line in lines:
#             pos = line.find(':')
#             goicuoc = clean_text(str(line[:pos]))
#             thongtin = clean_text(str(line[pos+1:]))
#             f_data.write("[\n")
#             f_data.write("\t\"thông tin gói cước {}\",\n".format(goicuoc))
#             f_data.write("\t\"{}\"\n".format(thongtin))
#             f_data.write("],\n")
# with open('viettel_giagoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     with open('viettel.json', 'a', encoding='utf-8') as f_data:
#         for line in lines:
#             pos = line.find(':')
#             goicuoc = clean_text(str(line[:pos]))
#             gia = clean_text(str(line[pos+1:]))
#             f_data.write("[\n")
#             f_data.write("\t\"giá gói cước {}\",\n".format(goicuoc))
#             f_data.write("\t\"{}\"\n".format(gia))
#             f_data.write("],\n")
# with open('viettel_dangkygoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     with open('viettel.json', 'a', encoding='utf-8') as f_data:
#         for line in lines:
#             pos = line.find(':')
#             goicuoc = clean_text(str(line[:pos]))
#             dangky = clean_text(str(line[pos+1:]))
#             f_data.write("[\n")
#             f_data.write("\t\"cú pháp đăng ký gói cước {}\",\n".format(goicuoc))
#             f_data.write("\t\"{}\"\n".format(dangky))
#             f_data.write("],\n")
# with open('viettel_huygiahangoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     with open('viettel.json', 'a', encoding='utf-8') as f_data:
#         for line in lines:
#             pos = line.find(':')
#             goicuoc = clean_text(str(line[:pos]))
#             huygiahan = clean_text(str(line[pos+1:]))
#             f_data.write("[\n")
#             f_data.write("\t\"cú pháp hủy gia hạn gói cước {}\",\n".format(goicuoc))
#             f_data.write("\t\"{}\"\n".format(huygiahan))
#             f_data.write("],\n")
# with open('viettel_huygoicuoc.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     with open('viettel.json', 'a', encoding='utf-8') as f_data:
#         for line in lines:
#             pos = line.find(':')
#             goicuoc = clean_text(str(line[:pos]))
#             huy = clean_text(str(line[pos+1:]))
#             f_data.write("[\n")
#             f_data.write("\t\"cú pháp hủy gói cước {}\",\n".format(goicuoc))
#             f_data.write("\t\"{}\"\n".format(huy))
#             f_data.write("],\n")


# train, test split file
# with open("viettel.json", 'r', encoding='utf-8') as f_train:
#     train_data = json.load(f_train)
# train, test = train_test_split(train_data, test_size=0.2)
# with open("train_finetune.json", 'w', encoding='utf-8') as f:
#     json.dump(train, f,  ensure_ascii=False, indent=4)
# with open("test_finetune.json", 'w', encoding='utf-8') as f:
#     json.dump(test, f,  ensure_ascii=False, indent=4)


# prompt
with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    with open('prompt.json', 'w', encoding='utf-8') as f_data:
        for line in lines:
            pos = line.find(':')
            goicuoc = clean_text(str(line[:pos]))
            thongtin = clean_text(str(line[pos+1:]))
            f_data.write("[\n")
            f_data.write("\t\"thông tin gói cước {}\",\n".format(goicuoc))
            f_data.write("\t\"gói cước {}, {}\",\n".format(goicuoc, thongtin))
            f_data.write("\t\"gói cước {}, {}\"\n".format(goicuoc, thongtin))
            f_data.write("],\n")
with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f_tt, open('viettel_giagoicuoc.txt', 'r', encoding='utf-8') as f_gia:
    lines_tt = f_tt.readlines()
    lines_gia = f_gia.readlines()
    dict_gia = {}
    dict_tt_gia = {}
    for line_gia in lines_gia:
        pos = line_gia.find(':')
        goicuoc = clean_text(str(line_gia[:pos]))
        gia = clean_text(str(line_gia[pos+1:]))
        dict_gia[goicuoc] = gia
    for line_tt in lines_tt:
        pos = line_tt.find(':')
        goicuoc = clean_text(str(line_tt[:pos]))
        thongtin = clean_text(str(line_tt[pos+1:]))
        if goicuoc in dict_gia.keys():
            dict_tt_gia[goicuoc] = thongtin
    with open('prompt.json', 'a', encoding='utf-8') as f_data:
        for goicuoc, thongtin in dict_tt_gia.items():
            f_data.write("[\n")
            f_data.write("\t\"giá gói cước {}\",\n".format(goicuoc))
            f_data.write("\t\"gói cước {}, {}\",\n".format(goicuoc, thongtin))
            f_data.write("\t\"{}\"\n".format(dict_gia[goicuoc]))
            f_data.write("],\n")
with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f_tt, open('viettel_dangkygoicuoc.txt', 'r', encoding='utf-8') as f_dk:
    lines_tt = f_tt.readlines()
    lines_dk = f_dk.readlines()
    dict_dk = {}
    dict_tt_dk = {}
    for line_dk in lines_dk:
        pos = line_dk.find(':')
        goicuoc = clean_text(str(line_dk[:pos]))
        dk = clean_text(str(line_dk[pos+1:]))
        dict_dk[goicuoc] = dk
    for line_tt in lines_tt:
        pos = line_tt.find(':')
        goicuoc = clean_text(str(line_tt[:pos]))
        thongtin = clean_text(str(line_tt[pos+1:]))
        if goicuoc in dict_dk.keys():
            dict_tt_dk[goicuoc] = thongtin
    with open('prompt.json', 'a', encoding='utf-8') as f_data:
        for goicuoc, thongtin in dict_tt_dk.items():
            f_data.write("[\n")
            f_data.write("\t\"cú pháp đăng ký gói cước {}\",\n".format(goicuoc))
            f_data.write("\t\"gói cước {}, {}\",\n".format(goicuoc, thongtin))
            f_data.write("\t\"{}\"\n".format(dict_dk[goicuoc]))
            f_data.write("],\n")
with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f_tt, open('viettel_huygiahangoicuoc.txt', 'r', encoding='utf-8') as f_hgh:
    lines_tt = f_tt.readlines()
    lines_hgh = f_hgh.readlines()
    dict_hgh = {}
    dict_tt_hgh = {}
    for line_hgh in lines_hgh:
        pos = line_hgh.find(':')
        goicuoc = clean_text(str(line_hgh[:pos]))
        hgh = clean_text(str(line_hgh[pos+1:]))
        dict_hgh[goicuoc] = hgh
    for line_tt in lines_tt:
        pos = line_tt.find(':')
        goicuoc = clean_text(str(line_tt[:pos]))
        thongtin = clean_text(str(line_tt[pos+1:]))
        if goicuoc in dict_hgh.keys():
            dict_tt_hgh[goicuoc] = thongtin
    with open('prompt.json', 'a', encoding='utf-8') as f_data:
        for goicuoc, thongtin in dict_tt_hgh.items():
            f_data.write("[\n")
            f_data.write("\t\"cú pháp hủy gia hạn gói cước {}\",\n".format(goicuoc))
            f_data.write("\t\"gói cước {}, {}\",\n".format(goicuoc, thongtin))
            f_data.write("\t\"{}\"\n".format(dict_hgh[goicuoc]))
            f_data.write("],\n")
with open('viettel_thongtingoicuoc.txt', 'r', encoding='utf-8') as f_tt, open('viettel_huygoicuoc.txt', 'r', encoding='utf-8') as f_h:
    lines_tt = f_tt.readlines()
    lines_h = f_h.readlines()
    dict_h = {}
    dict_tt_h = {}
    for line_h in lines_h:
        pos = line_h.find(':')
        goicuoc = clean_text(str(line_h[:pos]))
        h = clean_text(str(line_h[pos+1:]))
        dict_h[goicuoc] = h
    for line_tt in lines_tt:
        pos = line_tt.find(':')
        goicuoc = clean_text(str(line_tt[:pos]))
        thongtin = clean_text(str(line_tt[pos+1:]))
        if goicuoc in dict_h.keys():
            dict_tt_h[goicuoc] = thongtin
    with open('prompt.json', 'a', encoding='utf-8') as f_data:
        for goicuoc, thongtin in dict_tt_h.items():
            f_data.write("[\n")
            f_data.write("\t\"cú pháp hủy gói cước {}\",\n".format(goicuoc))
            f_data.write("\t\"gói cước {}, {}\",\n".format(goicuoc, thongtin))
            f_data.write("\t\"{}\"\n".format(dict_h[goicuoc]))
            f_data.write("],\n")