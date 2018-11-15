# -*- coding: utf-8 -*-
import xlrd
import xlwt
import argparse

out_book = xlwt.Workbook(encoding='utf-8', style_compression=0)
out_sheet = out_book.add_sheet('accuracy', cell_overwrite_ok=True)
column_index = 0
persons = 32
fold =10 
def fill_heaer(column_index,model_name,target_class):
	# table header
	out_sheet.write(0,column_index,"model_name:")
	out_sheet.write(0,column_index+1,model_name)
	out_sheet.write(1,column_index,"target_class:")
	out_sheet.write(1,column_index+1,target_class)
	out_sheet.write(2,column_index,"subject")
	out_sheet.write(2,column_index+1,"accuracy")

def fill_cells(dir_path,column_index,model_name,target_class):
	fill_heaer(column_index,model_name,target_class)
	total_accuracy = 0
	for sub in range(1,persons+1):
		subject = "s%02d"%sub
		accuracy = 0
		for count in range(fold):
			input_file = dir_path+target_class+"/"+str(subject)+"_"+str(count)+".xlsx"
			in_book = xlrd.open_workbook(input_file)
			sheet = in_book.sheet_by_name("condition")
			accuracy += sheet.cell_value(1,0)
		accuracy = (accuracy/10)*100
		total_accuracy += accuracy
		print(sub,":",accuracy)
		out_sheet.write(sub + 2,column_index, subject)
		out_sheet.write(sub + 2,column_index+1,accuracy)
	mean_accuracy = total_accuracy/persons
	print("mean accuracy:",mean_accuracy)
	out_sheet.write(sub+3,column_index,"mean:")
	out_sheet.write(sub+3,column_index+1,mean_accuracy)

def fill_cells_origin(dir_path,column_index,model_name,target_class):
	# table header
	fill_heaer(column_index,model_name,target_class)
	total_accuracy = 0
	for sub in range(1,persons+1):
		subject = "s%02d"%sub
		accuracy = 0
		input_file = dir_path+target_class+"/"+str(subject)+".xlsx"
		in_book = xlrd.open_workbook(input_file)
		sheet = in_book.sheet_by_name("condition")
		accuracy += sheet.cell_value(1,0)
		accuracy = (accuracy)*100
		total_accuracy += accuracy
		print(sub,":",accuracy)
		out_sheet.write(sub + 2,column_index, subject)
		out_sheet.write(sub + 2,column_index+1,accuracy)
	mean_accuracy = total_accuracy/persons
	print("mean accuracy:",mean_accuracy)
	out_sheet.write(sub+3,column_index,"mean:")
	out_sheet.write(sub+3,column_index+1,mean_accuracy)

if __name__ == '__main__':
	arousal_or_valence = "valence"
	fill_cells_origin("results/origin_",0,"without",arousal_or_valence)
	fill_cells("results/cv_",3,"with",arousal_or_valence)
	arousal_or_valence = "arousal"
	fill_cells_origin("results/origin_",6,"without",arousal_or_valence)
	fill_cells("results/cv_",9,"with",arousal_or_valence)

	out_book.save("accuracies.xls")