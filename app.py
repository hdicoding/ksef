"""KSEF 2027 | 네 가지 다중 팔레트 적재 방법 비교

계산 근거: KSEF2027_pallet_최종.ipynb의
① 높은 상자 우선, ② 낮은 상자 우선, ③ 높이 조합 우선,
④ 정수계획법(MILP)을 같은 상자 개수와 높이 제한으로 비교한다.

설치: python -m pip install "streamlit>=1.50,<2" "scipy>=1.11,<2" "plotly>=6,<7" "numpy>=1.26,<3"
실행: python -m streamlit run app.py

주의: 팔레트 1개에 독립적인 스택 4개, 쌓는 높이 2,330 mm를
기준으로 계산한다. 트럭 1대당 팔레트 18개는 도식용 가정이다.
실차의 적재 하중·안정성·결속·여유 치수는 검증되지 않았다.
"""

from functools import lru_cache
from html import escape
from math import ceil
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import streamlit as st

BOX_HEIGHTS = (220, 320, 420)
BOX_AREA = 600 * 400
PALLET_W, PALLET_D, PALLET_H = (1100, 1100, 150)
TRUCK_H = 2480
DEFAULT_HEIGHT = TRUCK_H - PALLET_H
SLOTS = 4
PALLETS_PER_TRUCK = 18
LAYOUT = ((50, 50, 600, 400), (650, 50, 400, 600),
          (450, 650, 600, 400), (50, 450, 400, 600))
# 3D 적재 구분용 색상: 인식은 유지하면서 채도를 낮춘 톤.
COLORS = {220: '#728E82', 320: '#BBA17B', 420: '#7A8EA6'}
METHODS = ('높은 상자 우선', '낮은 상자 우선', '높이 조합 우선', '최적 조합(MILP)')
METHOD_DESCRIPTIONS = {
    '높은 상자 우선': '높은 상자부터 들어갈 수 있는 첫 스택에 쌓기',
    '낮은 상자 우선': '낮은 상자부터 들어갈 수 있는 첫 스택에 쌓기',
    '높이 조합 우선': '남은 상자로 스택을 하나씩 가장 높게 쌓기',
    '최적 조합(MILP)': '모든 상자를 가장 적은 스택에 쌓기',
}


def validate(cargo, height):
    """잘못된 수량과 높이를 거부한다. bool도 정수로 취급하지 않는다."""
    try:
        cargo = tuple(cargo)
    except (TypeError, ValueError) as exc:
        raise ValueError('상자 3종의 개수를 입력하세요.') from exc
    if len(cargo) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in cargo):
        raise ValueError('상자 개수는 각각 0 이상의 정수여야 합니다.')
    if isinstance(height, bool) or not isinstance(height, Integral) or not (220 <= height <= DEFAULT_HEIGHT):
        raise ValueError('쌓는 높이는 220~2,330 mm 사이의 정수로 입력하세요.')
    return tuple(int(n) for n in cargo), int(height)


def total_height(counts):
    """한 스택에서 상자가 차지하는 높이(mm)."""
    return sum(n * h for n, h in zip(counts, BOX_HEIGHTS))


@lru_cache(maxsize=16)
def all_patterns(height):
    """노트북과 동일한, 한 스택에서 가능한 비어 있지 않은 모든 조합."""
    return tuple((a, b, c)
                 for a in range(height // 220 + 1)
                 for b in range(height // 320 + 1)
                 for c in range(height // 420 + 1)
                 if 0 < 220*a + 320*b + 420*c <= height)


@lru_cache(maxsize=16)
def ordered_patterns(height):
    """노트북과 같은 순서: 높이 → 상자 개수 → 420형 개수.

    세 값이 모두 같으면 같은 조합이므로 순서가 하나로 정해진다.
    """
    return tuple(sorted(all_patterns(height),
                        key=lambda p: (total_height(p), sum(p), p[2]),
                        reverse=True))


def first_fit(cargo, height, descending=True):
    """노트북의 첫 적합 스택 규칙. 상자를 하나씩 들어갈 수 있는 첫 번째 스택에 놓는다.

    계산량을 줄이기 위해 첫 적합 스택의 위치를 세그먼트 트리로 찾되,
    후보가 선택되는 순서는 노트북의 선형 탐색과 정확히 동일하다.
    """
    cargo, height = validate(cargo, height)
    total_boxes = sum(cargo)
    if total_boxes == 0:
        return []
    capacity = 1 << (total_boxes - 1).bit_length()
    tree = [height + 1] * (2 * capacity)
    filled = []
    stacks = []

    def put(j, used):
        k = capacity + j
        tree[k] = used
        k //= 2
        while k:
            tree[k] = min(tree[2*k], tree[2*k+1])
            k //= 2

    order = (2, 1, 0) if descending else (0, 1, 2)
    for index in order:
        h = BOX_HEIGHTS[index]
        if h > height and cargo[index]:
            raise ValueError(f'{h}형 상자가 입력 높이 제한 {height:,} mm를 초과합니다.')
        for _ in range(cargo[index]):
            target = height - h
            if tree[1] > target:
                p = [0, 0, 0]
                p[index] = 1
                stacks.append(p)
                filled.append(h)
                put(len(filled) - 1, h)
            else:
                j = 1
                while j < capacity:
                    j = j*2 if tree[j*2] <= target else j*2+1
                j -= capacity
                stacks[j][index] += 1
                filled[j] += h
                put(j, filled[j])
    return [tuple(p) for p in stacks]


def height_combo_first(cargo, height):
    """각 스택의 높이를 최대화하며 잔여 물량에 반복하는 탐욕 규칙."""
    cargo, height = validate(cargo, height)
    remaining = list(cargo)
    patterns = ordered_patterns(height)
    stacks = []
    while any(remaining):
        for pattern in patterns:
            if all(pattern[i] <= remaining[i] for i in range(3)):
                stacks.append(pattern)
                remaining = [remaining[i] - pattern[i] for i in range(3)]
                break
        else:
            raise ValueError('현재 높이 제한으로 적재할 수 없는 상자가 있습니다.')
    return stacks


def make_pallets(stacks):
    """노트북과 동일하게 스택을 높이순 정렬한 후 팔레트당 스택 네 개씩 묶는다."""
    stacks = sorted(stacks, key=lambda p: (total_height(p), p), reverse=True)
    return [tuple(stacks[i:i+SLOTS]) + ((0, 0, 0),) * (SLOTS - len(stacks[i:i+SLOTS]))
            for i in range(0, len(stacks), SLOTS)]


def solve_milp(cargo, height, time_limit=15):
    """노트북과 동일한 스택 수 최소화 정수계획법.

    하나의 최적화 결과가 실제 정수해, 전량 적재, 각 스택 높이 제한을
    만족하는지 확인하고, status==0인 경우만 최적성 '확인'으로 표시.
    """
    cargo, height = validate(cargo, height)
    if not any(cargo):
        return {'stacks': [], 'proved': True, 'status': '빈 물량', 'seconds': 0., 'lower_bound': 0.}
    patterns = all_patterns(height)
    valid = [p for p in patterns if all(p[i] <= cargo[i] for i in range(3))]
    if not valid:
        raise ValueError('현재 높이에서 전량 적재 가능한 조합이 없습니다.')
    matrix = np.asarray(valid, dtype=float).T
    cost = np.ones(len(valid), dtype=float)
    start = perf_counter()
    result = milp(
        c=cost,
        integrality=np.ones(len(valid), dtype=int),
        bounds=Bounds(np.zeros(len(valid)), np.full(len(valid), np.inf)),
        constraints=LinearConstraint(matrix, np.asarray(cargo, dtype=float), np.asarray(cargo, dtype=float)),
        options={'mip_rel_gap': 0.0, 'time_limit': time_limit},
    )
    seconds = perf_counter() - start
    bound = getattr(result, 'mip_dual_bound', None)
    if result.x is None:
        return {'stacks': None, 'proved': False,
                'status': f'MILP 실행가능해 없음 (solver 종료 {result.status})',
                'seconds': seconds, 'lower_bound': bound}
    rounded = np.rint(result.x)
    # 수치 오차를 허용하되 이산해와 정확한 입력 물량은 반드시 일치해야 한다.
    integer_ok = np.allclose(result.x, rounded, atol=1e-5, rtol=0)
    nonnegative_ok = np.all(rounded >= 0)
    cargo_ok = np.array_equal((matrix @ rounded).astype(np.int64), np.asarray(cargo, dtype=np.int64))
    if not (integer_ok and nonnegative_ok and cargo_ok):
        raise RuntimeError('정수계획법의 계산 결과가 입력 상자 수와 일치하지 않습니다.')
    stacks = [p for p, count in zip(valid, rounded.astype(int)) for _ in range(int(count))]
    return {'stacks': stacks, 'proved': result.status == 0,
            'status': '최적성 확인' if result.status == 0 else f'실행가능 배치(최적성 미확인; solver {result.status})',
            'seconds': seconds, 'lower_bound': bound}


def pallet_stats(cargo, height, stacks):
    """전량 적재·높이 제한 확인 후 팔레트 수 등의 공통 지표 산출."""
    if stacks is None:
        return None
    cargo, height = validate(cargo, height)
    if any(len(p) != 3 or any((n < 0 or int(n) != n) for n in p) for p in stacks):
        raise ValueError('잘못된 스택 조합이 있습니다.')
    if any(total_height(p) > height or total_height(p) <= 0 for p in stacks):
        raise ValueError('비어 있거나 높이 제한을 넘은 스택이 있습니다.')
    used = tuple(sum(p[k] for p in stacks) for k in range(3))
    if used != cargo:
        raise ValueError(f'상자 수량 불일치: 입력 {cargo}, 배치 {used}')
    pallets = make_pallets(stacks)
    last_volume = sum(total_height(p) for p in pallets[-1]) * BOX_AREA / 1e6 if pallets else 0.0
    total_volume = total_height(cargo) * BOX_AREA / 1e6
    utilization = 100 * total_volume * 1e6 / (len(pallets) * PALLET_W * PALLET_D * height) if pallets else 0.0
    return {'pallets': pallets, 'pallet_count': len(pallets), 'stack_count': len(stacks),
            'truck_count': ceil(len(pallets) / PALLETS_PER_TRUCK),
            'total_volume_l': total_volume, 'last_volume_l': last_volume,
            'utilization': utilization}


@st.cache_data(show_spinner=False, max_entries=32)
def calculate(cargo, height):
    """네 방법을 같은 입력으로 계산한다. 증명 실패를 최적값으로 취급하지 않는다."""
    cargo, height = validate(cargo, height)
    too_high = [h for h, n in zip(BOX_HEIGHTS, cargo) if n and h > height]
    if too_high:
        raise ValueError(f'{", ".join(map(str, too_high))}형 상자가 쌓는 높이를 초과합니다.')
    plans = {
        METHODS[0]: first_fit(cargo, height, True),
        METHODS[1]: first_fit(cargo, height, False),
        METHODS[2]: height_combo_first(cargo, height),
    }
    milp_result = solve_milp(cargo, height)
    plans[METHODS[3]] = milp_result['stacks']
    metrics = {name: pallet_stats(cargo, height, stacks) for name, stacks in plans.items()}
    if milp_result['proved'] and metrics[METHODS[3]] is not None:
        optimum = metrics[METHODS[3]]['pallet_count']
        if any(metric and metric['pallet_count'] < optimum for metric in metrics.values()):
            raise RuntimeError('검증 오류: 최적 조합보다 적은 팔레트 수를 발견했습니다.')
    return metrics, milp_result


def pattern_to_stack(pattern):
    """3D 그림에 쓰는 상자 높이 배열. 위쪽과 아래쪽 색 배열은 지표에 영향 없음."""
    return [h for i, h in reversed(list(enumerate(BOX_HEIGHTS))) for _ in range(pattern[i])]


def counts_for_pallet(pallet):
    return tuple(sum(column[k] for column in pallet) for k in range(3))


def compact_table(headers, rows):
    head = ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
    body = ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows)
    return '<table class="plan-table"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>'

def whole_load_figure(pallets, height, *, first_pallet_no=1, chart_height=475, zoom=1.0):
    """한 대의 트럭과 실제 상자 배치를 독립적인 3D 장면에 그린다.

    zoom은 카메라 거리만 바꾸며 트럭/상자 실제 치수와 적재 계산은 바꾸지 않는다.
    상단의 트럭과 마지막 트럭은 별개 Figure라 확대가 서로 영향을 주지 않는다.
    """
    if not 1 <= len(pallets) <= PALLETS_PER_TRUCK:
        raise ValueError('차량 그림에는 1~18개의 팔레트를 전달해야 합니다.')
    if zoom <= 0:
        raise ValueError('화면 확대 비율은 양수여야 합니다.')
    import math
    import plotly.graph_objects as go

    # 대량 입력에서도 그림 객체 수가 늘지 않도록 색상별로 면을 모은다.
    batches = {}
    edges = {'x': [], 'y': [], 'z': []}
    triangles = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    edge_pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                  (0,4),(1,5),(2,6),(3,7)]

    def add_box(x, y, z, w, d, h, color, description):
        batch = batches.setdefault(color, {'x': [], 'y': [], 'z': [],
                                            'i': [], 'j': [], 'k': [], 'text': []})
        points = [(x,y,z),(x+w,y,z),(x+w,y+d,z),(x,y+d,z),
                  (x,y,z+h),(x+w,y,z+h),(x+w,y+d,z+h),(x,y+d,z+h)]
        offset = len(batch['x'])
        for px, py, pz in points:
            batch['x'].append(px)
            batch['y'].append(py)
            batch['z'].append(pz)
            batch['text'].append(description)
        for a, b, c in triangles:
            batch['i'].append(offset+a)
            batch['j'].append(offset+b)
            batch['k'].append(offset+c)
        for a, b in edge_pairs:
            for dimension, index in [('x',0),('y',1),('z',2)]:
                edges[dimension].extend([points[a][index], points[b][index], None])

    wing_edges = {'x': [], 'y': [], 'z': []}
    badges = {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': []}
    ink = {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': []}

    def face(store, points, text=None):
        offset = len(store['x'])
        for x,y,z in points:
            store['x'].append(x)
            store['y'].append(y)
            store['z'].append(z)
            if 'text' in store:
                store['text'].append(text or '')
        store['i'].extend([offset,offset])
        store['j'].extend([offset+1,offset+2])
        store['k'].extend([offset+2,offset+3])

    def number_plate(number, x, y, z, reverse=False):
        """팔레트 측면의 번호판을 확대해 작은 글씨가 뭉개지지 않도록 한다.

        번호는 3D 모델의 실제 팔레트 번호를 사용한다. 팔레트 폭(1,100mm)
        안에서 번호판 폭을 제한하며, 번호판 하단은 팔레트 밑면과 일치시킨다.
        """
        def rect(store, left, bottom, width, tall, depth):
            # 반대쪽 면에서 볼 때도 글자 방향이 바르게 보이도록 반전한다.
            corners = [(left, bottom), (left + width, bottom),
                       (left + width, bottom + tall), (left, bottom + tall)]
            face(store, [(x - px if reverse else x + px, y + depth, z + pz)
                         for px, pz in corners])

        digits = str(number)
        # 기존 번호판(높이 120mm, 글자 약 90mm)을 약 2.5배 확대한다.
        # 한 자릿수~여러 자릿수 번호가 모두 같은 높이로 표시된다.
        scale = 2.55
        digit_step = 46
        plate_height = 282
        plate_width = max(320, len(digits) * digit_step * scale + 90)
        plate_width = min(plate_width, 900)  # 팔레트 한 개의 전면 폭 이하
        direction = 1 if reverse else -1
        rect(badges, -plate_width / 2, 0, plate_width, plate_height,
             direction * 14)

        segments = {
            'a': (5, 84, 28, 8), 'b': (33, 49, 8, 35),
            'c': (33, 14, 8, 35), 'd': (5, 6, 28, 8),
            'e': (-3, 14, 8, 35), 'f': (-3, 49, 8, 35),
            'g': (5, 45, 28, 8),
        }
        digit_segments = [
            'abcdef', 'bc', 'abdeg', 'abcdg', 'bcfg',
            'acdfg', 'acdefg', 'abc', 'abcdefg', 'abcdfg',
        ]
        start = -len(digits) * digit_step * scale / 2 + 4 * scale
        for index, digit in enumerate(digits):
            for segment in digit_segments[int(digit)]:
                dx, dz, w, h = segments[segment]
                rect(ink,
                     start + (index * digit_step + dx) * scale,
                     (dz + 10) * scale, w * scale, h * scale,
                     direction * 22)

    def wing_line(points):
        for dimension, index in [('x', 0), ('y', 1), ('z', 2)]:
            wing_edges[dimension].extend([p[index] for p in points] + [None])

    def wing_panel(points, description):
        # 적재물이 가려지지 않도록 날개 면은 생략하고 외곽선만 표시한다.
        wing_line(points + [points[0]])

    # 도식용 적재함: 10,020×2,400 mm에 1,100 mm 팔레트를 2줄×9개 배치.
    # 무게와 작업 여유를 계산한 운송 계획은 아니며, 기존 팔레트 결과를 시각화한다.
    # 한 Figure에는 트럭 1대만 넣는다. 차량 간 거리 때문에 축소되는 현상 방지.
    truck_count = 1
    deck_top = 900
    wheels = []

    for truck in range(truck_count):
        tx, ty = 0, 0
        add_box(tx,ty,deck_top-160,10020,2400,160,'#9EABB0','트럭 적재함 바닥')
        add_box(tx+300,ty+450,450,9400,1500,290,'#43525B','차대')
        add_box(tx-1750,ty+80,500,1650,2240,1900,'#E3E9E9','운전석')
        add_box(tx-1610,ty+60,1450,1050,20,660,'#456575','운전석 측면 창')
        add_box(tx-1770,ty+230,1420,20,1940,700,'#597C8C','앞 유리')
        add_box(tx-1830,ty+100,470,160,2200,190,'#687980','앞 범퍼')
        for axle in (-800,7200,8500):
            for side in (-80,2200):
                wheels.append((tx+axle,ty+side))
        roof_z = deck_top+2480
        # 적재함의 모서리 지지대와 지붕 테두리.
        for end in (0,9960):
            for side in (0,2340):
                add_box(tx+end,ty+side,deck_top,60,60,2480,'#ADBFC5','윙바디 프레임')
        add_box(tx,ty+1170,roof_z,10020,60,80,'#879EA9','지붕 중앙 경첩')
        # 닫혔을 때 ㄱ자인 '지붕 절반 + 옆문'을 중앙 경첩에서 함께 회전한다.
        # 두 면의 길이와 직각을 유지하여 펼친 날개의 연결 구조를 표현한다.
        angle = math.radians(75)
        for side in (-1,1):
            hinge_y = ty+1200
            hinge_z = roof_z+80
            shoulder_y = hinge_y + side*1200*math.cos(angle)
            shoulder_z = hinge_z + 1200*math.sin(angle)
            outer_y = shoulder_y + side*2480*math.sin(angle)
            outer_z = shoulder_z - 2480*math.cos(angle)
            wing_panel([(tx,hinge_y,hinge_z),(tx+10020,hinge_y,hinge_z),
                        (tx+10020,shoulder_y,shoulder_z),(tx,shoulder_y,shoulder_z)],
                       '들어 올린 지붕')
            wing_panel([(tx,shoulder_y,shoulder_z),(tx+10020,shoulder_y,shoulder_z),
                        (tx+10020,outer_y,outer_z),(tx,outer_y,outer_z)],
                       '지붕에 연결된 옆문')
            # 앞뒤 지지대만 남겨 상자 위에 선이 복잡하게 겹치지 않게 한다.
            for end in (80,9940):
                wing_line([(tx+end,ty+(40 if side<0 else 2360),roof_z-700),
                           (tx+end,(shoulder_y+outer_y)/2,(shoulder_z+outer_z)/2)])


    for position, stacks in enumerate(pallets):
        # 실제 전체 적재표의 팔레트 번호를 유지한다(예: 37~41번).
        number = first_pallet_no + position
        tx, ty = 0, 0
        column, row = divmod(position, 2)
        x0, y0 = tx+60+column*1100, ty+100+row*1100
        add_box(x0,y0,deck_top,PALLET_W,PALLET_D,PALLET_H,'#BCA784',f'{number}번 팔레트')
        for slot, ((x,y,w,d), stack) in enumerate(zip(LAYOUT,stacks),1):
            z = deck_top+PALLET_H
            for level,h in enumerate(stack,1):
                add_box(x0+x,y0+y,z,w,d,h,COLORS[h],
                        f'{number}번 팔레트 · {slot}번 스택 · 아래에서 {level}번째 · {h}형')
                z += h
        # 앞줄은 가까운 옆면, 뒷줄은 반대 옆면에 번호판을 붙인다.
        number_plate(number,x0+550,y0 if row==0 else y0+PALLET_D,
                     deck_top+20,reverse=row==1)

    fig = go.Figure()
    for color, batch in batches.items():
        fig.add_trace(go.Mesh3d(**batch,color=color,flatshading=True,
                               lighting=dict(ambient=.8,diffuse=.6),
                               hovertemplate='%{text}<extra></extra>',showscale=False))
    fig.add_trace(go.Scatter3d(**edges,mode='lines',line=dict(color='white',width=1),
                              hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter3d(**wing_edges,mode='lines',
                              line=dict(color='#AAB8BF',width=2),hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(**badges,color='#234C39',lighting=dict(ambient=1,diffuse=0),
                           hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(**ink,color='#FFFFFF',lighting=dict(ambient=1,diffuse=0),
                           hoverinfo='skip'))
    # 바퀴는 차체 폭 방향의 원통으로 그린다.
    for x,y in wheels:
        angles = np.linspace(0,2*np.pi,17)[:-1]
        xs = np.tile(x+350*np.cos(angles),2)
        ys = np.repeat([y,y+280],16)
        zs = np.tile(350+350*np.sin(angles),2)
        faces = []
        for j in range(16):
            k = (j+1)%16
            faces.extend([(j,k,j+16),(k,k+16,j+16)])
        for j in range(1,15):
            faces.extend([(0,j+1,j),(16,16+j,16+j+1)])
        a,b,c = zip(*faces)
        fig.add_trace(go.Mesh3d(x=xs,y=ys,z=zs,i=a,j=b,k=c,color='#354047',
                               flatshading=True,hoverinfo='skip',showscale=False))

    # 독립된 3D 차트는 같은 기준 치수로 구성한다.
    # 카메라를 조절할 수 있는 것은 마지막 차량뿐이며 몸체 비율은 동일하다.
    xr, yr, zr = (-2150, 10350), (-1750, 4250), (-220, 4740)
    # 확대 배율은 카메라 거리로만 제어한다. 동일한 차량 치수를 유지하고
    # 폭이 좁은 작은 차트에서는 zoom<1로 바깥쪽이 잘리지 않도록 한다.
    eye = dict(x=0.92 / zoom, y=-1.50 / zoom, z=1.06 / zoom)
    spans = [xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]]
    scale = 1.9/max(spans)
    axis = dict(visible=False)
    fig.update_layout(
        height=chart_height, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', uirevision=f'wing-v6-{first_pallet_no}-{height}',
        scene=dict(
            xaxis=dict(**axis,range=xr), yaxis=dict(**axis,range=yr), zaxis=dict(**axis,range=zr),
            aspectmode='manual', aspectratio=dict(zip(('x','y','z'),[s*scale for s in spans])),
            camera=dict(eye=eye, projection=dict(type='orthographic')),
        ),
    )
    return fig





STYLE = """
<style>
/* 전체 톤은 중립색 위주로 정리하고 강조색은 선택한 버튼에만 사용한다. */
.stApp {background:#F7F8F6;color:#29352F;}
header[data-testid="stHeader"] {background:transparent;height:0;}
[data-testid="stToolbar"], [data-testid="stMainMenu"] {display:none;}
.block-container {max-width:1600px;padding:7px 18px 12px;}
[data-testid="stVerticalBlock"] {gap:4px;}
.app-banner {background:#344F40;color:#FFFFFF;border-radius:10px;
    padding:11px 18px;margin:0 0 5px;text-align:center;}
.app-banner h1 {font-size:29px;line-height:1.3;font-weight:750;
    color:white;margin:0;padding:0;letter-spacing:-.4px;}
.compare-title {font-size:14px;font-weight:750;color:#2D3A32;
    margin:0 0 5px;text-align:center;}

/* 네 가지 적재 방법은 전부 동일한 중립색. 선택한 항목만 진한 녹색이다. */
div[class*="st-key-method_option_"] {background:#FFFFFF;border-radius:9px;
    padding:7px 9px 8px!important;}
div[class*="st-key-method_option_"] [data-testid="stVerticalBlock"] {gap:3px!important;}
div[class*="st-key-method_option_"] button {min-height:37px!important;
    height:37px!important;border-radius:7px!important;transition:background .15s ease;}
div[class*="st-key-method_option_"] button p {font-size:13px;font-weight:750;line-height:1.2;}
div[class*="st-key-method_option_"] button[kind="secondary"] {
    background:#F7F8F7!important;border:1px solid #D8DDD9!important;
    color:#37483E!important;}
div[class*="st-key-method_option_"] button[kind="secondary"]:hover {
    background:#EFF2EF!important;border-color:#B4C0B7!important;}
div[class*="st-key-method_option_"] button[kind="primary"] {
    background:#3A604A!important;border:1px solid #3A604A!important;
    color:#FFFFFF!important;}
div[class*="st-key-method_option_"] button[kind="primary"] p {color:#FFFFFF!important;}
.method-help {font-size:11px;color:#778079;line-height:1.35;
    text-align:center;margin:0;}
/* 수량 3개는 줄바꿈 없이 한 줄에서 비교한다. */
.method-metrics {display:flex;justify-content:space-between;align-items:baseline;
    gap:3px;font-size:10px;color:#67726B;padding:2px 0 0;
    flex-wrap:nowrap;white-space:nowrap;min-width:0;
    font-variant-numeric:tabular-nums;}
.method-metrics > span {white-space:nowrap;flex:0 0 auto;}
.method-metrics strong {font-size:12px;color:#2B3830;margin-left:2px;font-weight:750;}
.method-metrics .last-volume {font-size:10px;color:#667169;}
.method-metrics .last-volume strong {font-size:11px;font-weight:650;margin-left:2px;}
@media(max-width:1100px) {
  .method-metrics {font-size:9px;gap:2px;}
  .method-metrics strong {font-size:11px;}
  .method-metrics .last-volume,.method-metrics .last-volume strong {font-size:9px;}
}

/* 상단 입력: 장식용 가로색 선 제거, 흰색 카드와 동일한 회색 테두리 적용. */
div[class*="st-key-input_panel"] {background:#F0F2EF;
    border:1px solid #E1E5E0;border-radius:11px;padding:9px 11px 10px!important;
    margin-bottom:5px;}
div[class*="st-key-input_panel"] [data-testid="stHorizontalBlock"] {align-items:stretch;}
div[class*="st-key-input_card_"] {height:100%;min-width:0;background:#FFFFFF;
    border:1px solid #E0E5E0;border-radius:8px;padding:8px 10px!important;}
div[class*="st-key-input_card_"] [data-testid="stVerticalBlock"] {gap:5px!important;}
div[class*="st-key-input_card_"] [data-testid="stNumberInput"] label {margin:0!important;}
div[class*="st-key-input_card_"] [data-testid="stNumberInput"] label p,
.input-detail-label {font-size:12px!important;font-weight:700!important;
    color:#46544A!important;line-height:1.25!important;margin:0!important;}
div[class*="st-key-input_card_"] [data-baseweb="input"] {background:#FAFBFA!important;
    border:1px solid #E1E5E1!important;border-radius:7px!important;box-shadow:none!important;}
div[class*="st-key-input_card_"] [data-baseweb="input"]:focus-within {
    border-color:#7C9A85!important;box-shadow:0 0 0 1px #7C9A85!important;}
div[class*="st-key-input_card_"] [data-testid="stNumberInput"] input {
    font-size:15px!important;font-weight:650!important;color:#2C3831!important;
    padding-top:7px!important;padding-bottom:7px!important;min-width:0!important;}
div[class*="st-key-input_card_"] [data-testid="stNumberInput"] button {
    min-width:32px!important;color:#526156!important;}
div[class*="st-key-input_card_details"] button[kind="secondary"] {
    border:1px solid #D1D9D1!important;background:#F5F7F4!important;
    color:#3E5446!important;border-radius:7px!important;min-height:39px!important;
    height:39px!important;font-weight:750!important;}
div[class*="st-key-input_card_details"] button[kind="secondary"]:hover {
    background:#E9EEE9!important;border-color:#B4C2B6!important;}
div[class*="st-key-input_card_details"] button p {font-size:13px!important;font-weight:750!important;}
@media(max-width:900px) {
  div[class*="st-key-input_panel"] {padding:7px!important;}
  div[class*="st-key-input_card_"] {padding:6px!important;}
  div[class*="st-key-input_card_"] [data-testid="stNumberInput"] button {min-width:26px!important;}
  div[class*="st-key-input_card_"] [data-testid="stNumberInput"] input {font-size:13px!important;}
}
.color-key {display:flex;justify-content:center;gap:20px;font-size:12px;margin-top:-3px;}
/* 위쪽 고정 차량: 기본 컴포넌트 여백을 줄여 같은 행의 간격을 일정하게 유지. */
div[class*="st-key-fixed_trucks_row"] [data-testid="stVerticalBlock"] {gap:0!important;}
/* 위쪽에 다섯 대일 때는 캔버스 경계를 12px씩 포개어 잘림을 줄인다.
   실제 차량/적재물이 겹치는 것이 아니라 표시 영역만 겹친다. */
div[class*="st-key-fixed_trucks_row_5"] [data-testid="stHorizontalBlock"] {
    gap:0!important;overflow:visible!important;
}
div[class*="st-key-fixed_trucks_row_5"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width:0!important;overflow:visible!important;
}
div[class*="st-key-fixed_trucks_row_5"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:not(:first-child) {
    margin-left:-12px!important;
}
div[class*="st-key-last_truck_row"] [data-testid="stVerticalBlock"] {gap:0!important;}
/* 트럭 번호는 그림 아래에 배치한다. 번호만 녹색으로 강조해 복잡한 장식을 피한다. */
.truck-label,.truck-last-label {
    display:flex;align-items:baseline;justify-content:center;flex-wrap:wrap;
    column-gap:7px;row-gap:1px;text-align:center;line-height:1.35;
    margin:-3px 0 6px;font-variant-numeric:tabular-nums;
}
.truck-label .truck-number {color:#315942;font-size:14px;font-weight:800;white-space:nowrap;}
.truck-last-label .truck-number {color:#315942;font-size:16px;font-weight:800;white-space:nowrap;}
.truck-label .truck-pallets,.truck-last-label .truck-pallets {
    color:#66736A;font-size:11px;font-weight:600;white-space:nowrap;
}
@media(max-width:1080px) {
  .truck-label .truck-number {font-size:12px;}
  .truck-label .truck-pallets {font-size:10px;}
}

.plan-table {border-collapse:collapse;table-layout:fixed;width:100%;font-size:12px;
    background:white;font-variant-numeric:tabular-nums;}
.plan-table th,.plan-table td {padding:6px 5px;text-align:center;
    border-bottom:1px solid #E9ECE9;white-space:nowrap;}
.plan-table th {background:#F0F2EF;position:sticky;top:0;font-weight:600;}
.overview-scroll {max-height:530px;overflow:auto;border:1px solid #E2E6E1;
    border-radius:8px;background:white;}
.detail-scroll {max-height:65vh;overflow:auto;border:1px solid #E2E6E1;
    border-radius:8px;background:white;}
.detail-table {width:100%;table-layout:fixed;border-collapse:collapse;
    background:white;font-size:12px;font-variant-numeric:tabular-nums;}
.detail-table th,.detail-table td {padding:4px 6px;line-height:20px;
    text-align:center;white-space:nowrap;border-bottom:1px solid #E9ECE9;}
.detail-table th {position:sticky;top:0;background:#F0F2EF;font-weight:600;}
.detail-table tr.pallet-start td {border-top:2px solid #D9DEDA;}
div[role="dialog"]:has(.detail-scroll) {width:min(680px,94vw)!important;max-width:680px!important;}
@media(max-width:780px) {
  .block-container {padding:8px 10px;}
  .app-banner {padding:10px 12px;}
  .app-banner h1 {font-size:24px;}
  .method-help {font-size:10px;}
  .method-metrics {font-size:10px;gap:2px;}
}
</style>
"""



def truck_display_plan(total_trucks):
    """화면 배치: 꽉 찬 트럭 최대 5대, 마지막 1대는 아래 중앙에 표시.

    반환: (윗줄 트럭 번호, 마지막 트럭 번호, 생략된 중간 트럭 수)
    """
    if total_trucks < 1:
        raise ValueError('트럭 수는 1 이상이어야 합니다.')
    fixed = list(range(1, min(total_trucks, 6)))
    omitted = (total_trucks - 1) - len(fixed)
    return fixed, total_trucks, omitted


def main():
    st.set_page_config(page_title='팔레트 적재 방법 비교', page_icon='📦',
                       layout='wide', initial_sidebar_state='collapsed')
    st.html(STYLE)
    st.html('<div class="app-banner"><h1>팔레트 적재 방법 비교</h1></div>')

    # 상단 입력은 한 줄의 얇은 패널에 동일한 높이의 다섯 칸으로 정렬한다.
    # 각 st.number_input의 기본 +/- 버튼과 즉시 재계산 동작은 유지한다.
    with st.container(key='input_panel'):
        fields = st.columns([1, 1, 1, 1.14, .67], gap='small', vertical_alignment='bottom')
        quantities = []
        for i, h in enumerate(BOX_HEIGHTS):
            with fields[i]:
                with st.container(key=f'input_card_{h}'):
                    quantities.append(st.number_input(
                        f'{h}형 · 개수', min_value=0, max_value=10000,
                        value=100, step=1, key=f'qty_{h}',
                    ))
        cargo = tuple(quantities)
        with fields[3]:
            with st.container(key='input_card_height'):
                height = st.number_input(
                    '쌓는 높이 · mm', min_value=220,
                    max_value=DEFAULT_HEIGHT, value=DEFAULT_HEIGHT,
                    step=10, key='height',
                )
        with fields[4]:
            with st.container(key='input_card_details'):
                st.html('<div class="input-detail-label">상세 보기</div>')
                detail_clicked = st.button('적재표', width='stretch', key='details')

    if not any(cargo):
        st.info('상자 수량을 입력하면 네 가지 방법의 적재 결과가 즉시 계산됩니다.')
        return
    try:
        with st.spinner('네 가지 방법의 배치를 계산하는 중…'):
            metrics, milp_info = calculate(cargo, height)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    # MILP의 유효해가 없을 경우에도 세 규칙의 결과를 보여주되
    # 네 번째 방법이 최적이라고 잘못 표시하지 않는다.
    if st.session_state.get('method') not in METHODS:
        st.session_state.method = METHODS[3]
    if metrics[st.session_state.method] is None:
        st.session_state.method = METHODS[2]
    method = st.session_state.method
    selected = metrics[method]
    optimal = metrics[METHODS[3]]
    proof = milp_info['proved'] and optimal is not None
    n_pal = selected['pallet_count']

    def select_method(name):
        st.session_state.method = name

    # 과도한 상단 수치 요약과 최적성 완료 문구는 표시하지 않는다.
    # 단, MILP가 최적성을 증명하지 못했다면 연구 결과의 오해를 막기 위해 경고한다.
    if not proof:
        st.warning(f'정수계획법: {milp_info["status"]}. 최소 팔레트 수가 확인되지 않았습니다.')

    @st.dialog('팔레트별 상세 적재표', width='small')
    def show_details():
        rows = []
        for pallet_no, pallet in enumerate(selected['pallets'], 1):
            for slot, pattern in enumerate(pallet, 1):
                values = [f'{pallet_no}-{slot}', *pattern, height-total_height(pattern)]
                cells = ''.join(f'<td>{escape(str(value))}</td>' for value in values)
                cls = 'pallet-start' if slot == 1 else ''
                rows.append(f'<tr class="{cls}">{cells}</tr>')
        headers = ['팔레트-스택', '220형', '320형', '420형', '여유 mm']
        head = ''.join(f'<th>{escape(h)}</th>' for h in headers)
        st.html('<div class="detail-scroll"><table class="detail-table">'
                f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    if detail_clicked:
        show_details()

    comparison, picture, totals = st.columns([1.15, 3.55, 1.30], gap='small')
    with comparison:
        st.html('<div class="compare-title">적재 방법 비교</div>')
        for index, name in enumerate(METHODS):
            stat = metrics[name]
            with st.container(border=True, key=f'method_option_{index}'):
                # 화면은 간결한 명칭을 사용하고 실제 내부 계산은 MILP를 유지한다.
                display_name = '최적 조합' if name == METHODS[3] else name
                st.button(display_name, key=f'method_pick_{index}',
                          type='primary' if method == name else 'secondary',
                          disabled=stat is None, width='stretch',
                          on_click=select_method, args=(name,))
                st.html(f'<div class="method-help">{escape(METHOD_DESCRIPTIONS[name])}</div>')
                if stat is None:
                    st.caption('계산 가능한 적재 배치가 없습니다.')
                else:
                    # 주요 지표 옆에 마지막 팔레트의 적재 부피를 작게 표시한다.
                    st.html('<div class="method-metrics">'
                            f'<span>팔레트<strong>{stat["pallet_count"]:,}개</strong></span>'
                            f'<span>스택<strong>{stat["stack_count"]:,}개</strong></span>'
                            f'<span class="last-volume">마지막 팔레트<strong>{stat["last_volume_l"]:,.0f} L</strong></span>'
                            '</div>')
    with picture:
        # 꽉 찬 트럭은 최대 다섯 대까지 같은 줄에 일정 간격으로 배치한다.
        # 마지막 트럭은 별도의 아래 행에 표시하고, 이 트럭만 회전·확대할 수 있다.
        # 고정형 그래프에도 Plotly WebGL 렌더링을 유지해야 보이지 않는 문제가 없다.
        total_trucks = ceil(n_pal / PALLETS_PER_TRUCK)
        fixed_trucks, last_truck, omitted_trucks = truck_display_plan(total_trucks)

        def draw_truck(truck_number, *, chart_height, last=False, zoom=1.0):
            # 팔레트 18개를 1대에 배치하되 실제 입력 및 조합 결과를 그대로 사용한다.
            start = (truck_number - 1) * PALLETS_PER_TRUCK
            end = min(start + PALLETS_PER_TRUCK, n_pal)
            real_pallets = selected['pallets'][start:end]
            stacks = [[pattern_to_stack(p) for p in pallet] for pallet in real_pallets]
            label_class = 'truck-last-label' if last else 'truck-label'
            truck_name = f'{truck_number:,}번 트럭'
            if last and total_trucks > 1:
                truck_name = '마지막 ' + truck_name

            fig = whole_load_figure(
                stacks, height, first_pallet_no=start + 1,
                chart_height=chart_height, zoom=zoom,
            )
            if not last:
                # 그림 자체는 마지막 트럭과 똑같은 3D 렌더러를 사용한다.
                # 장면의 드래그 모드만 끄므로 WebGL 렌더링은 유지된다.
                fig.update_layout(scene_dragmode=False, hovermode=False)
            st.plotly_chart(
                fig, width='stretch', key=f'truck_view_{truck_number}',
                config={
                    'displaylogo': False,
                    'displayModeBar': False,
                    'scrollZoom': last,
                    'doubleClick': False if not last else 'reset',
                    'responsive': True,
                },
            )
            # 그림 아래에 차량 번호를 크게, 팔레트 수량은 보조 정보로 작게 표시한다.
            st.html(
                f'<div class="{label_class}">'
                f'<span class="truck-number">{escape(truck_name)}</span>'
                f'<span class="truck-pallets">팔레트 {len(real_pallets):,}개</span>'
                '</div>'
            )

        if fixed_trucks:
            # 열마다 별도 Figure를 사용해 트럭들 사이를 12~16px 정도 띄운다.
            # 다섯 대까지 한 행에 나란히 보이도록 차량 수별 화면 배율을 조정한다.
            # 마지막 차량은 이 행에 넣지 않는다.
            fixed_count = len(fixed_trucks)
            sizes = {1: (190, 0.65), 2: (177, 0.60), 3: (165, 0.58),
                     4: (156, 0.56), 5: (150, 0.50)}
            full_height, full_zoom = sizes[fixed_count]
            with st.container(key=f'fixed_trucks_row_{fixed_count}'):
                if fixed_count == 1:
                    # 트럭 두 대인 경우 꽉 찬 트럭을 위쪽 중앙에 작게 배치.
                    _, central, _ = st.columns([0.24, 1, 0.24], gap='small')
                    with central:
                        draw_truck(fixed_trucks[0], chart_height=full_height,
                                   zoom=full_zoom)
                else:
                    cols = st.columns([1] * fixed_count, gap=None if fixed_count == 5 else 'small')
                    for truck_number, col in zip(fixed_trucks, cols):
                        with col:
                            draw_truck(truck_number, chart_height=full_height,
                                       zoom=full_zoom)

        # 마지막 트럭은 위쪽의 고정 트럭 수와 관계없이 아래쪽 중앙에 배치한다.
        # 회전·확대 조작은 이 Figure에만 허용한다.
        with st.container(key='last_truck_row'):
            _, center, _ = st.columns([0.07, 1, 0.07], gap='small')
            with center:
                draw_truck(last_truck,
                           chart_height=425 if total_trucks == 1 else 305,
                           last=True,
                           zoom=0.89 if total_trucks == 1 else 0.84)

        if omitted_trucks:
            st.caption(f'화면에는 처음 5대와 마지막 트럭을 표시합니다. '
                       f'중간 {omitted_trucks:,}대는 오른쪽 적재표에서 확인할 수 있습니다.')

        st.html('<div class="color-key"><span style="color:#526F61">■ 220형</span>'
                '<span style="color:#97784D">■ 320형</span>'
                '<span style="color:#5F7590">■ 420형</span></div>')
    with totals:
        st.html('<div class="compare-title">팔레트별 상자 수</div>')
        rows = [[f'{i}번', *counts_for_pallet(pallet)]
                for i, pallet in enumerate(selected['pallets'], 1)]
        st.html('<div class="overview-scroll">' + compact_table(
            ['팔레트', '220형', '320형', '420형'], rows) + '</div>')


if __name__ == '__main__':
    main()
