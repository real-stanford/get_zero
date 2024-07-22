colors = \
"""\
154, 38, 38
221, 221, 13
11, 82, 57
3, 64, 120
90, 59, 114"""

colors = colors.split('\n')
colors = [[int(z)/255 for z in y.split(', ')] for y in colors]

link_names_lst = [['palm'], ['mcp', 'pip_4'], ['pip', 'thumb_pip'], ['dip', 'thumb_dip'], ['fingertip', 'thumb_fingertip']]

for color, link_names in zip(colors, link_names_lst):
    color = [str(x) for x in color]
    for link_name in link_names:
        print(f'<material name="{link_name}">\n    <color rgba="{" ".join(color)} 1.0" />\n</material>')
