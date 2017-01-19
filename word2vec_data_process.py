import os
import numpy as np

ORI_FILE_NAME = 'honghua_mars_p13n_1d_new_cate_query_impression_20161204'
INPUT_FILE_PATH = '../data/' + ORI_FILE_NAME
OUTPUT_FILE_PATH = '../data/' + ORI_FILE_NAME + '_processed'

AD_NUMBER_THERSHOLD = 3

def main():
	# check file existence
	if not os.path.exists(INPUT_FILE_PATH):
		print 'file does not exist: %s'%INPUT_FILE_PATH
		return

	dataForWordVec = []
	querySet = set()
	goodsIdSet = set()
	with open(INPUT_FILE_PATH, 'r') as inputData:
		print 'parsing file: %s'%(INPUT_FILE_PATH)
		parseInfo = {
		'total': 0,
		'parsed': 0,
		'failed': 0,
		'filted': 0
		}
		for item in inputData:
			parseInfo['total'] += 1
			isSuccess, query, goodsId = parse_one_line(item)
			if not isSuccess:
				parseInfo['failed'] += 1
				continue
			if len(goodsId) < AD_NUMBER_THERSHOLD:
				parseInfo['filted'] += 1
			dataForWordVec.append([query,goodsId])
			querySet.add(query)
			goodsIdSet |= set(goodsId)
			parseInfo['parsed'] += 1

		print 'parsed info: total: %d, parsed: %d, filted: %d, failed: %d,'%(parseInfo['total'], parseInfo['parsed'], parseInfo['filted'], parseInfo['failed'])
		print 'query set: %d, goods_id set: %d'%(len(querySet), len(goodsIdSet))

	with open(OUTPUT_FILE_PATH, 'w') as outFile:
		print 'results output to file: %s'%(OUTPUT_FILE_PATH)
		for item in dataForWordVec:
			outFile.write(' '.join(item[1]))
		print 'writing done'


def parse_one_line(item):
	if not item:
		return False, '', []

	splitedSet = item.split('')
	if len(splitedSet) <= 2:
		return False, '', []

	query = splitedSet[1]
	adSet = splitedSet[3:]
	goodsId = []
	for ad in adSet:
		adInfo = ad.split("_")
		goodsId.append(adInfo[3])

	return True, query, goodsId

if __name__ == '__main__':
	main()

