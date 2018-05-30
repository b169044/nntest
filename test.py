#-*- coding: UTF-8 -*-
# https://blog.csdn.net/m0_37650263/article/details/77343220
# https://blog.csdn.net/leiting_imecas/article/details/71246541

import csv

out = open('result2.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')
csv_write.writerow(['22'])

