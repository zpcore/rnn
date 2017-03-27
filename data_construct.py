import tensorflow as tf

class data_construct:
	data=[]
	length=0
	count=0
	num_example=0;
	def __init__(self, length, data):
		self.data=data
		self.length=length
		self.num_example=len(data)

	def next_batch(batch_size):
		
		temp_data=data.range(count*(length+batch_size-1),(count+1)*(length+batch_size-1)-1)
		batch_x[:]=[]
		for i in batch_size:
			for j in length:
				batch_x.append(temp_data[i*length+j][0:1,0])
			batch_y.append(temp_data[length-1][2,0])
		count+=1#count+=length
		if (count+length-1)*batch_size>=len(data):
			count=0
		#batch_y=temp_data[length-1][2:3,0]
		
		return batch_x, batch_y


	@property
	def data(self):
		return self.data

	@property
	def count(self):
		return self.count

	@property
	def num_example():
		return self.num_example