# _*_ coding: utf-8 _*_
#---
# I'm going to study Python from now on.
#---
#
# It's been modified to learn the github.
#
#[文字列の連結 1]
test_str = 'python'
test_str = test_str + '-'
test_str = test_str + 'izm'
print '      '
print test_str
#
#[文字列の連結 2]
test_str2 = '012'
test_str2 += '345'
test_str2 += '678'
print '      '
print test_str2
#
#[数値を文字列に変換]
test_integer = 100
print '     '
print str(test_integer) + '円'
print '     '
#
#[文字列の置換]
test_str3 = 'python_izm'
print test_str3.replace('izm', 'ism')
print '     '
#
#[文字列の分割]
print test_str3.split('_')
print '     '
#
print test_str2.rjust(12, '$')
print '     '
#
#[文字列が任意の文字で始まっているか調べる]
#['python'で始まっているか？]
print test_str3.startswith('python')
#['izm'で始まっているか？]
print test_str3.startswith('izm')
print '     '
#
#[文字列に任意の文字が含まれているか]
#[文字列に'z'が含まれているか？]
print 'z' in test_str3
#[文字列に's'が含まれているか？]
print 's' in test_str3
print '     '
print test_str3
print '     '
#
#[大文字・小文字変換]
test_str3 = 'kosei-halfmoon.dyndns-ip.com'
print test_str3.upper()
print test_str3.lower()
print '     '
#
#[文字列の先頭または末尾を削除した値を取得する]
print '---------------------------------'
test_str3 = '   kosei-halfmoon.dyndns-ip.com'
print test_str3
#
test_str3 = test_str3.lstrip()
print test_str3
print '     '
#
test_str3 = test_str3.lstrip('kosei-halfmoon.')
print test_str3
print '     '
#
print '---------------------------------'
print '     '
test_str3 = 'kosei-halfmoon.dyndns-ip.com    '
print test_str3 + '/'
print '     '
#
test_str3 = test_str3.rstrip()
print test_str3 + '/'
print '     '
#
test_str3 = test_str3.rstrip('.com')
print test_str3
print '     '
#
ask_name = """
+--------------------+
+ あなたのお名前は？ +
+--------------------+
"""
print ask_name

your_name = raw_input('>> ')

print u'                    '
print u'こんにちは、{} さん！' .format(your_name)
print u'                    '
