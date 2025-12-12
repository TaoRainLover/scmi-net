from transformers import BertTokenizer,BertModel, BertConfig, RobertaModel, RobertaTokenizer, RobertaConfig


model_name = '../pretained_model/roberta-base'
config = RobertaConfig.from_pretrained(model_name)	# 这个方法会自动从官方的s3数据库下载模型配置、参数等信息（代码中已配置好位置）
tokenizer = RobertaTokenizer.from_pretrained(model_name)	 # 这个方法会自动从官方的s3数据库读取文件下的vocab.txt文件
model = RobertaModel.from_pretrained(model_name).to('cuda')		# 这个方法会自动从官方的s3数据库下载模型信息
print(model)


text = "I don't understand why this is so complicated for people when they get here. It's just a simple form. I just need an ID."
# text = "I don't know if I ever can forgive you for that. Why'd you wait so long? I sat in my room wondering if I was crazy for thinking about you."
text = "also I was the point person on my company's transition from the KL-5 to GR-6 system."
encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
# move to cuda
# for key in encoded_input.keys():
# 	encoded_input[key] = encoded_input[key].cuda()
output = model(**encoded_input)

print(output['last_hidden_state'].shape)