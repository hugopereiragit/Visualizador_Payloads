#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
import ipywidgets as widgets
from IPython.display import display, HTML
import logging

# -------------------------------------------------------------------------------------
# GLOBAL PARAMETERS (Configurable via Widgets)
# -------------------------------------------------------------------------------------

# Truck parameters:
comprimento_input_caminhao = 13.0      # Comprimento interno do caminhão (m)
largura_input_caminhao     = 2.4       # Largura interna do caminhão (m)
altura_input_caminhao      = 2.7       # Altura interna do caminhão (m)
capacidade_peso_caminhao   = 100000    # Capacidade máxima de peso (kg)

# Painel Principal (Main Panel) parameters:
peso_painel_principal = 50

# Pod parameters:
peso_pod_vazio = 500

# Painel Pequeno (Small Panel) defaults:
num_paineis_pequenos = None   
altura_painel_pequeno = None  
espacamento_entre_paineis_pequenos = 0.02  

# Variáveis para os Painéis Principais:
num_paineis_principais = None
largura_painel_principal = None

# Outras variáveis (serão definidas via widgets):
comprimento_input_pod = None
largura_input_pod = None
altura_base_pod = None
espessura_chao_pod = None
altura_input_total_pod = None
espessura_borda_pod = None
altura_painel_principal = None
espacamento_entre_paineis_principais = None
espessura_painel_principal = None
largura_input_painel_pequeno = None
comprimento_input_painel_pequeno = None
altura_input_toilet = None
largura_input_toilet = None
comprimento_input_toilet = None
altura_input_sink = None
largura_input_sink = None
comprimento_input_sink = None

# Tolerance for comparing floats:
tol = 1e-6

# -------------------------------------------------------------------------------------
# FUNCTIONS: DRAWING & CALCULATIONS 
# -------------------------------------------------------------------------------------

def desenhar_cubo(ax, x, y, z, comp, larg, alt, cor='cyan', alpha=0.3, remove_top_bottom=False):
    vertices = [
        [x,         y,         z],
        [x + comp,  y,         z],
        [x + comp,  y + larg,  z],
        [x,         y + larg,  z],
        [x,         y,         z + alt],
        [x + comp,  y,         z + alt],
        [x + comp,  y + larg,  z + alt],
        [x,         y + larg,  z + alt]
    ]
    if remove_top_bottom:
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]]
        ]
    else:
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[2], vertices[3]]
        ]
    poly = Poly3DCollection(faces, facecolors=cor, linewidths=0.5,
                             edgecolors=(0, 0, 0, 0.3), alpha=alpha)
    ax.add_collection3d(poly)

def calcular_max_paineis_por_pod(largura_input_pod, espessura_borda_pod, main_panel_thickness, main_panel_spacing,
                                 num_paineis_pequenos, small_panel_thickness, small_panel_spacing, largura_input_toilet):
    # If num_paineis_pequenos is not set, use a default of 1
    if num_paineis_pequenos is None:
        num_paineis_pequenos = 1
    largura_interna = largura_input_pod - 2 * espessura_borda_pod
    espaco_necessario_pequenos = (small_panel_spacing +
                                  num_paineis_pequenos * small_panel_thickness +
                                  (num_paineis_pequenos - 1) * small_panel_spacing)
    espaco_disponivel_small = largura_interna - espaco_necessario_pequenos
    max_from_small = int((espaco_disponivel_small + main_panel_spacing) // (main_panel_thickness + main_panel_spacing))

    vertical_clearance = largura_input_pod - 2 * espessura_borda_pod - largura_input_toilet
    max_from_cube = int((vertical_clearance + main_panel_spacing) // (main_panel_thickness + main_panel_spacing))

    return min(max_from_small, max_from_cube)

def calcular_peso_pod_com_paineis(num_paineis, peso_pod_base, peso_um_painel):
    return peso_pod_base + (num_paineis * peso_um_painel)

def calcular_num_paineis_por_perimetro_altura(comprimento_pod, largura_pod, altura_painel):
    perimetro = 2 * (comprimento_pod + largura_pod)
    num_paineis = perimetro / altura_painel  
    print(f"Dado painéis com altura de {altura_painel:.6f}m e um pod com {perimetro:.6f}m de perímetro, precisamos de {num_paineis:.6f} painéis.")
    return num_paineis

def empilhar_pods_no_caminhao(num_paineis):
    num_pods_por_comprimento = int(comprimento_input_caminhao // comprimento_input_pod)
    num_pods_por_largura     = int(largura_input_caminhao // largura_input_pod)
    num_pods_por_altura      = int(altura_input_caminhao // altura_input_total_pod)
    pods_colocados = []
    peso_total = 0
    for h in range(num_pods_por_altura):
        for l in range(num_pods_por_largura):
            for c in range(num_pods_por_comprimento):
                peso_pod = calcular_peso_pod_com_paineis(num_paineis, peso_pod_vazio, peso_painel_principal)
                if peso_total + peso_pod <= capacidade_peso_caminhao:
                    x = c * comprimento_input_pod
                    y = l * largura_input_pod
                    z = h * altura_input_total_pod
                    pods_colocados.append((x, y, z, num_paineis))
                    peso_total += peso_pod
                else:
                    print("Capacidade máxima de peso do caminhão atingida.")
                    return pods_colocados
    return pods_colocados

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [x_limits[1] - x_limits[0],
              y_limits[1] - y_limits[0],
              z_limits[1] - z_limits[0]]
    max_range = max(ranges)
    x_mid = (x_limits[0] + x_limits[1]) * 0.5
    y_mid = (y_limits[0] + y_limits[1]) * 0.5
    z_mid = (z_limits[0] + z_limits[1]) * 0.5
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

def desenhar_paineis_pequenos(ax, x, y, z):
    x_inicial = x + espessura_borda_pod
    small_panel_right = x_inicial + comprimento_input_painel_pequeno  
    right_border = x + comprimento_input_pod - espessura_borda_pod
    sink_x0 = right_border - (comprimento_input_toilet + comprimento_input_sink)
    if small_panel_right > sink_x0:
        print("AVISO: Os painéis pequenos estão tocando os cubos (toilet/sink)!")
    y_interno = y + espessura_borda_pod
    ultima_painel_y = y_interno + (num_paineis_principais * (espessura_painel_principal + espacamento_entre_paineis_principais)) - espacamento_entre_paineis_principais
    espaco_restante = (y + largura_input_pod - espessura_borda_pod) - ultima_painel_y
    # print(f"Espaço remanescente para painéis pequenos: {espaco_restante:.3f}m")
    espaco_necessario = (espacamento_entre_paineis_pequenos +
                          num_paineis_pequenos * largura_input_painel_pequeno +
                          (num_paineis_pequenos - 1) * espacamento_entre_paineis_pequenos)
    if espaco_necessario >= espaco_restante:
        print(f"AVISO: O espaço necessário para {num_paineis_pequenos} painéis pequenos ({espaco_necessario:.3f}m) excede o espaço disponível ({espaco_restante:.3f}m).")
        return(0)
    else:
        desenhar_cubo(ax, x_inicial, ultima_painel_y, z,
                      comprimento_input_painel_pequeno,
                      espacamento_entre_paineis_pequenos,
                      altura_painel_pequeno,
                      cor='red', alpha=0.8)
        current_y = ultima_painel_y + espacamento_entre_paineis_pequenos
        for i in range(num_paineis_pequenos):
            if current_y + largura_input_painel_pequeno <= y + largura_input_pod - espessura_borda_pod:
                desenhar_cubo(ax, x_inicial, current_y, z,
                              comprimento_input_painel_pequeno,
                              largura_input_painel_pequeno,
                              altura_painel_pequeno,
                              cor='green', alpha=1)
                if i < num_paineis_pequenos - 1:
                    desenhar_cubo(ax, x_inicial, current_y + largura_input_painel_pequeno, z,
                                  comprimento_input_painel_pequeno,
                                  espacamento_entre_paineis_pequenos,
                                  altura_painel_pequeno,
                                  cor='red', alpha=0.8)
                current_y += largura_input_painel_pequeno + espacamento_entre_paineis_pequenos
            else:
                print("O painel pequeno ultrapassaria os limites do pod. Parando a adição.")
                break

def desenhar_toilet_sink_relative_x(ax, pod_x, pod_y, pod_z):
    right_border = pod_x + comprimento_input_pod - espessura_borda_pod
    top_border   = pod_y + largura_input_pod - espessura_borda_pod
    toilet_x0 = right_border - comprimento_input_toilet
    toilet_y0 = top_border - largura_input_toilet
    sink_x0 = toilet_x0 - comprimento_input_sink
    sink_y0 = top_border - largura_input_sink
    desenhar_cubo(ax, toilet_x0, toilet_y0, pod_z,
                  comprimento_input_toilet, largura_input_toilet, altura_input_toilet,
                  cor='orange', alpha=1)
    desenhar_cubo(ax, sink_x0, sink_y0, pod_z,
                  comprimento_input_sink, largura_input_sink, altura_input_sink,
                  cor='yellow', alpha=1)

def desenhar_toilet_sink_relative_y(ax, pod_x, pod_y, pod_z):
    # Calculate the same right and top borders.
    right_border = pod_x + comprimento_input_pod - espessura_borda_pod
    top_border   = pod_y + largura_input_pod - espessura_borda_pod
    # In this orientation, both units remain flush on the right,
    # but they are stacked vertically along the Y axis.

    # The toilet remains at the top.
    toilet_x0 = right_border - comprimento_input_toilet
    toilet_y0 = top_border - largura_input_toilet

    # The sink is placed directly below the toilet.
    sink_x0 = right_border - comprimento_input_sink
    sink_y0 = toilet_y0 - espessura_borda_pod - largura_input_sink

    desenhar_cubo(ax, toilet_x0, toilet_y0, pod_z,
                  comprimento_input_toilet, largura_input_toilet, altura_input_toilet,
                  cor='orange', alpha=1)
    desenhar_cubo(ax, sink_x0, sink_y0, pod_z,
                  comprimento_input_sink, largura_input_sink, altura_input_sink,
                  cor='yellow', alpha=1)

def desenhar_pod_unico(num_paineis):
    # Check if the available horizontal space for main panels is sufficient.
    if altura_painel_principal > (comprimento_input_pod - espessura_borda_pod):
        print("AVISO: A altura dos painéis é maior que o espaço existente dentro do pod")
        return 0
    largura_interna = largura_input_pod - 2 * espessura_borda_pod
    total_ocupado = num_paineis * espessura_painel_principal + (num_paineis - 1) * espacamento_entre_paineis_principais
    if total_ocupado > largura_interna:
        print(f"AVISO: Os painéis principais e espaçamentos ocupam {total_ocupado}m, excedendo a largura interna do pod ({largura_interna}m).")
        return 0

    x, y, z = 0, 0, 0
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(12, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the outer pod shell.
    desenhar_cubo(ax, x, y, z, 
                  comprimento_input_pod, largura_input_pod, altura_input_total_pod,
                  cor='cyan', alpha=0.5, remove_top_bottom=True)

    # Draw the inner cavity to represent the hollow space.
    inner_x = x + espessura_borda_pod
    inner_y = y + espessura_borda_pod
    inner_z = z + espessura_chao_pod  
    inner_comp = comprimento_input_pod - 2 * espessura_borda_pod
    inner_larg = largura_input_pod - 2 * espessura_borda_pod
    inner_alt = altura_input_total_pod - espessura_chao_pod
    # Use a low alpha and remove top/bottom to reveal the interior.
    desenhar_cubo(ax, inner_x, inner_y, inner_z,
                  inner_comp, inner_larg, inner_alt,
                  cor='white', alpha=0.2, remove_top_bottom=True)

    # Set the starting positions for the main panels using the inner cavity.
    x_interno = inner_x
    y_interno = inner_y
    for i in range(num_paineis):
        painel_x = x_interno
        painel_y = y_interno + i * (espessura_painel_principal + espacamento_entre_paineis_principais)
        painel_z = z
        desenhar_cubo(ax, painel_x, painel_y, painel_z,
                      altura_painel_principal, espessura_painel_principal, largura_painel_principal,
                      cor='gray', alpha=1)
        if i < num_paineis - 1:
            gap_x = x_interno
            gap_y = painel_y + espessura_painel_principal
            gap_z = z
            desenhar_cubo(ax, gap_x, gap_y, gap_z,
                          altura_painel_principal, espacamento_entre_paineis_principais, largura_painel_principal,
                          cor='red', alpha=0.8)

    desenhar_paineis_pequenos(ax, x, y, z)
    if (comprimento_input_toilet < available_length and
        comprimento_input_sink < available_length and
        (largura_input_sink + largura_input_toilet + espessura_borda_pod * 2) < largura_input_pod):
        desenhar_toilet_sink_relative_y(ax, x, y, z)
    else:
        desenhar_toilet_sink_relative_x(ax, x, y, z)



    ax.set_xticks(np.arange(0, comprimento_input_caminhao + 0.1, 0.5))
    ax.set_yticks(np.arange(0, largura_input_caminhao + 0.1, 0.5))
    ax.set_zticks(np.arange(0, altura_input_caminhao + 0.1, 0.5))
    ax.set_xlabel("Comprimento (m)", fontsize=12)
    ax.set_ylabel("Largura (m)", fontsize=12)
    ax.set_zlabel("Altura (m)", fontsize=12)
    ax.set_title("Visualização 3D de um Único Pod", fontsize=14)
    ax.view_init(elev=90, azim=-90)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

def main_invertido(num_paineis):

    if altura_painel_principal > (comprimento_input_pod - espessura_borda_pod):
        print("AVISO: A altura dos painéis é maior que o espaço existente dentro do pod")
        return 0

    if altura_painel_pequeno > altura_input_caminhao:
        print("AVISO: A altura dos painéis pequenos é maior que a altura interna do caminhão.")
        return 0

    if altura_painel_principal > altura_input_caminhao: 
        print("AVISO: A altura dos painéis é maior que a altura interna do caminhão.")
        return 0

    if altura_input_toilet > altura_input_caminhao:
        print("AVISO: A altura do toilet é maior que a altura interna do caminhão.")
        return 0

    if altura_input_sink > altura_input_caminhao:
        print("AVISO: A altura do sink é maior que a altura interna do caminhão.")
        return 0


    largura_interna = largura_input_pod - 2 * espessura_borda_pod
    total_ocupado = num_paineis * espessura_painel_principal + (num_paineis - 1) * espacamento_entre_paineis_principais
    if total_ocupado > largura_interna:
        print(f"AVISO: Os painéis principais e espaçamentos ocupam {total_ocupado}m, excedendo a largura interna do pod ({largura_interna}m).")
        return 0

    max_paineis_possivel = calcular_max_paineis_por_pod(
        largura_input_pod, espessura_borda_pod,
        espessura_painel_principal, espacamento_entre_paineis_principais,
        num_paineis_pequenos, largura_input_painel_pequeno,
        espacamento_entre_paineis_pequenos, largura_input_toilet
    )

    #legacy code to check if the number of panels exceeds the maximum allowed
    #if num_paineis > max_paineis_possivel:
    #    print(f"AVISO: O número de painéis desejado ({num_paineis}) excede o máximo ({max_paineis_possivel}) que cabe no pod.")
    #    return 0

    pods = empilhar_pods_no_caminhao(num_paineis)
    print(f"Dado painéis com espessura de {espessura_painel_principal}m, espaçamento de {espacamento_entre_paineis_principais}m e borda de {espessura_borda_pod}m:")

    #Legacy code to check the number of panels needed to cover the perimeter
    #print(f"Número máximo de painéis permitidos: {max_paineis_possivel}")
    #minimo_paineis = calcular_num_paineis_por_perimetro_altura(comprimento_input_pod, largura_input_pod, largura_painel_principal)
    #print(f"Número mínimo de painéis necessários para cobrir o perímetro: {minimo_paineis}")


    print(f"Largura do painel escolhido: {num_paineis}")
    print(f"Total de pods acomodados: {len(pods)}")

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    desenhar_cubo(ax, x=0, y=0, z=0,
                  comp=comprimento_input_caminhao,
                  larg=largura_input_caminhao,
                  alt=altura_input_caminhao,
                  cor='blue', alpha=0.00)

    for (pod_x, pod_y, pod_z, num_paineis) in pods:
        # Draw the pod boundary.
        desenhar_cubo(ax, pod_x, pod_y, pod_z,
                      comprimento_input_pod, largura_input_pod, altura_input_total_pod,
                      cor='cyan', alpha=0.2, remove_top_bottom=True)

        # Set the internal starting positions using the pod wall thickness.
        x_interno = pod_x + espessura_borda_pod
        y_interno = pod_y + espessura_borda_pod

        # Draw the main panels and gaps.
        for i in range(num_paineis):
            painel_x = x_interno
            painel_y = y_interno + i * (espessura_painel_principal + espacamento_entre_paineis_principais)
            painel_z = pod_z
            desenhar_cubo(ax, painel_x, painel_y, painel_z,
                          altura_painel_principal, espessura_painel_principal, largura_painel_principal,
                          cor='gray', alpha=1)
            if i < num_paineis - 1:
                gap_x = x_interno
                gap_y = painel_y + espessura_painel_principal
                gap_z = pod_z
                desenhar_cubo(ax, gap_x, gap_y, gap_z,
                              altura_painel_principal, espacamento_entre_paineis_principais, largura_painel_principal,
                              cor='red', alpha=0.8)
        # Draw the small panels and sink/toilet.
        desenhar_paineis_pequenos(ax, pod_x, pod_y, pod_z)
        if (comprimento_input_toilet < available_length and
            comprimento_input_sink < available_length and
            (largura_input_sink + largura_input_toilet + espessura_borda_pod * 2) < largura_input_pod):
            desenhar_toilet_sink_relative_y(ax, pod_x, pod_y, pod_z)
        else:
            desenhar_toilet_sink_relative_x(ax, pod_x, pod_y, pod_z)

    ax.set_xticks(np.arange(0, comprimento_input_caminhao + 0.1, 0.5))
    ax.set_yticks(np.arange(0, largura_input_caminhao + 0.1, 0.5))
    ax.set_zticks(np.arange(0, altura_input_caminhao + 0.1, 0.5))

    ax.set_xlabel("Comprimento (m)", fontsize=12)
    ax.set_ylabel("Largura (m)", fontsize=12)
    ax.set_zlabel("Altura (m)", fontsize=12)
    ax.set_title("Visualização 3D com Pods Empilhados", fontsize=14)

    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

# --- FUNÇÕES PARA OTIMIZAÇÃO DOS PAINÉIS ---
def otimizar_solucao_main_panels_exact():
    """
    Organic optimization for main panels.
    Generates candidate solutions (n, h) using the pod’s perimeter.
    - Filters candidates yielding at least 2 floors.
    - First tries candidates with 3 floors; if none, then candidates with 2 floors.
    - In the 2-floor case, if no candidate has an exact fit (remainder==0),
      we now choose the candidate with the highest n (i.e. maximum allowed panels)
      so that with default values the candidate with 8 panels (h = 0.95) is picked.
    Returns a tuple (n, h, floors, remainder).
    """
    solucoes = encontrar_solucoes_altura_painel(comprimento_input_pod, largura_input_pod, altura_minima=0.2)
    max_paineis_possivel = calcular_max_paineis_por_pod(
        largura_input_pod, espessura_borda_pod,
        espessura_painel_principal, espacamento_entre_paineis_principais,
        num_paineis_pequenos, largura_input_painel_pequeno,
        espacamento_entre_paineis_pequenos, largura_input_toilet
    )
    solucoes_filtradas = [(n, h) for (n, h) in solucoes if n <= max_paineis_possivel]
    candidates = []
    for (n, h) in solucoes_filtradas:
        total_height = h + altura_base_pod + espessura_chao_pod
        floors = math.floor(altura_input_caminhao / total_height)
        remainder = altura_input_caminhao - floors * total_height
        if floors >= 2:
            candidates.append((n, h, floors, remainder))

    # First, try candidates with 3 floors
    candidates_3 = [cand for cand in candidates if cand[2] == 3]
    if candidates_3:
        exact_candidates = [cand for cand in candidates_3 if abs(cand[3]) < tol]
        if exact_candidates:
            optimal = min(exact_candidates, key=lambda x: x[0])
        else:
            optimal = min(candidates_3, key=lambda x: x[3])
    else:
        # For candidates with 2 floors, choose the one with the maximum number of panels
        # (i.e. highest n) if no exact candidate exists.
        candidates_2 = [cand for cand in candidates if cand[2] == 2]
        if candidates_2:
            exact_candidates = [cand for cand in candidates_2 if abs(cand[3]) < tol]
            if exact_candidates:
                optimal = min(exact_candidates, key=lambda x: x[0])
            else:
                optimal = max(candidates_2, key=lambda x: x[0])
        else:
            print("Nenhum candidato válido encontrado para os painéis principais.")
            return None

    print(f"Sol. Painéis Principais: n = {optimal[0]}, h = {optimal[1]:.6f}m, floors = {optimal[2]}, remainder = {optimal[3]:.6f}m")
    return optimal

def otimizar_solucao_small_panels():
    """
    Organic optimization for small panels.
    Uses the optimal main panel candidate to determine available space.
    Here, we compute available space based on the pod’s comprimento minus the sink/toilet:
         available_space = comprimento_input_pod - (comprimento_input_sink + comprimento_input_toilet)
    And for each small panel candidate (n, h) computed as h = comprimento_input_pod / n,
    we require h <= available_space.
    Also, we require that the small panel option yields at least as many truck floors
    as the main panel option.
    Returns a tuple (n_small, h_small, floors_small, remainder_small).
    """
    optimal_main = otimizar_solucao_main_panels_exact()
    if optimal_main is None:
        print("Não foi possível otimizar os painéis pequenos pois a otimização dos principais falhou.")
        return None
    n_main, h_main, floors_main, _ = optimal_main

    # Use pod comprimento and sink/toilet to compute available space (horizontal dimension)
    available_space = comprimento_input_pod - (comprimento_input_sink + comprimento_input_toilet)
    # For candidate small panels, we use: h_small = comprimento_input_pod / n_small
    solucoes = encontrar_solucoes_small_panels(comprimento_input_pod)
    candidates = []
    for (n_small, h_small) in solucoes:
        # Here, total_small is simply the candidate panel width.
        total_small = h_small
        if total_small <= available_space:
            total_height_small = h_small + altura_base_pod + espessura_chao_pod
            floors_small = math.floor(altura_input_caminhao / total_height_small)
            if floors_small >= floors_main:
                remainder_small = available_space - total_small
                candidates.append((n_small, h_small, floors_small, remainder_small))

    if candidates:
        exact_candidates = [cand for cand in candidates if abs(cand[3]) < tol]
        if exact_candidates:
            optimal = min(exact_candidates, key=lambda x: x[0])
        else:
            optimal = min(candidates, key=lambda x: x[3])
        return optimal
    else:
        print("Nenhuma opção válida encontrada para os painéis pequenos.")
        return None

# --- FUNÇÕES DE SELEÇÃO VIA WIDGETS ---
def selecionar_solucao_small_panels():
    """
    Exibe a solução otimizada para os painéis pequenos e permite a seleção manual.
    """
    print("\nCalculando a solução otimizada para os painéis pequenos (Teto)...")
    #sol_otima = otimizar_solucao_small_panels()
    print("\nOpções para seleção manual:")
    solucoes = plot_solucoes_small_panels(comprimento_input_pod)
    options = [(f"{n} painéis, altura: {h:.6f}m", (n, h)) for n, h in solucoes]
    dropdown = widgets.Dropdown(options=options, description="Solução Painéis Pequenos (Teto):")
    button = widgets.Button(description="Selecionar", layout=widgets.Layout(width='150px'))
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            output.clear_output()
            global num_paineis_pequenos, altura_painel_pequeno, altura_input_total_pod
            n_escolhido, h_escolhido = dropdown.value
            num_paineis_pequenos = n_escolhido
            altura_painel_pequeno = h_escolhido
            altura_input_total_pod = altura_painel_pequeno + altura_base_pod + espessura_chao_pod
            print(f"Solução para painéis pequenos (Teto) selecionada: {n_escolhido} painéis com largura {h_escolhido:.6f}m")
            selecionar_solucao_widget()

    button.on_click(on_button_clicked)
    display(widgets.VBox([dropdown, button, output]))

def selecionar_solucao_widget():
    """
    Exibe as opções para os painéis principais e, ao selecionar, atualiza os parâmetros e redesenha as visualizações.
    """
    solucoes_filtradas = plot_solucoes_altura_painel(comprimento_input_pod, largura_input_pod, altura_minima=0.2)
    options = [(f"{n} painéis, largura: {h:.6f}m", (n, h)) for n, h in solucoes_filtradas]
    dropdown = widgets.Dropdown(options=options, description="Solução Painéis Principais:")
    button = widgets.Button(description="Selecionar", layout=widgets.Layout(width='150px'))
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            output.clear_output()
            n_escolhido, h_escolhido = dropdown.value
            print(f"Solução selecionada para painéis principais: {n_escolhido} painéis com largura {h_escolhido:.6f}m")
            global largura_painel_principal, num_paineis_principais, altura_input_total_pod
            num_paineis_principais = n_escolhido
            largura_painel_principal = h_escolhido
            if altura_input_total_pod < (largura_painel_principal + altura_base_pod + espessura_chao_pod):
                altura_input_total_pod = largura_painel_principal + altura_base_pod + espessura_chao_pod
            desenhar_pod_unico(num_paineis_principais)
            main_invertido(num_paineis_principais)

    button.on_click(on_button_clicked)
    display(widgets.VBox([dropdown, button, output]))

# -------------------------------------------------------------------------------------
# FUNÇÕES PARA SOLUÇÕES E PLOTAGEM
# -------------------------------------------------------------------------------------

def encontrar_solucoes_altura_painel(comprimento_pod, largura_pod, altura_minima=0.2):
    """
    Encontra soluções para a altura dos painéis principais dado o perímetro do pod.
    """
    perimetro = 2 * (comprimento_pod + largura_pod)
    solucoes = []
    n_max = int(perimetro / altura_minima)
    for n in range(1, n_max + 1):
        h = perimetro / n
        if h >= altura_minima:
            solucoes.append((n, round(h, 6)))
    return solucoes

def plot_solucoes_altura_painel(comprimento_pod, largura_pod, altura_minima=0.2):
    """
    Plota as soluções candidatas para os painéis principais e destaca a opção ótima.
    """
    solucoes = encontrar_solucoes_altura_painel(comprimento_pod, largura_pod, altura_minima)
    if(comprimento_input_toilet < available_length and
       comprimento_input_sink < available_length and
       (largura_input_sink + largura_input_toilet + espessura_borda_pod * 2) < largura_input_pod):
                max_paineis_possivel = calcular_max_paineis_por_pod(
                    largura_input_pod, espessura_borda_pod,
                    espessura_painel_principal, espacamento_entre_paineis_principais,
                    num_paineis_pequenos, largura_input_painel_pequeno,
                    espacamento_entre_paineis_pequenos, 0
        )
    else:   

        if(largura_input_sink > largura_input_toilet):
                max_paineis_possivel = calcular_max_paineis_por_pod(
                    largura_input_pod, espessura_borda_pod,
                    espessura_painel_principal, espacamento_entre_paineis_principais,
                    num_paineis_pequenos, largura_input_painel_pequeno,
                    espacamento_entre_paineis_pequenos, largura_input_sink
                    )
        else:
                max_paineis_possivel = calcular_max_paineis_por_pod(
                    largura_input_pod, espessura_borda_pod,
                    espessura_painel_principal, espacamento_entre_paineis_principais,
                    num_paineis_pequenos, largura_input_painel_pequeno,
                    espacamento_entre_paineis_pequenos, largura_input_toilet
                    )
    solucoes_filtradas = [(n, h) for (n, h) in solucoes if n <= max_paineis_possivel]

    # Determine optimal candidate: pelo menos 2 pisos e menor h.
    optimal_index = None
    optimal_candidate = None
    for idx, (n, h) in enumerate(solucoes_filtradas):
        total_height = h + altura_base_pod + espessura_chao_pod
        floors = math.floor(altura_input_caminhao / total_height)
        if floors < 2:
            continue
        if (optimal_candidate is None) or (h < optimal_candidate[1]):
            optimal_candidate = (n, h, floors)
            optimal_index = idx

    # Extract data: x will be número de painéis, y will be largura do painel.
    num_paineis_lista = [n for n, h in solucoes_filtradas]
    larguras = [h for n, h in solucoes_filtradas]
    perimetro = 2 * (comprimento_pod + largura_pod)
    # For the marker, x is max number of panels and y is its corresponding width
    h_max = perimetro / max_paineis_possivel

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(num_paineis_lista, larguras, color='blue', label="Soluções")
    ax.plot(num_paineis_lista, larguras, color='blue', linestyle='--', alpha=0.6)
    ax.scatter(max_paineis_possivel, h_max, color='red', s=100,
               label=f"Máximo de painéis = {max_paineis_possivel}")
    ax.set_xlabel("Número de Painéis")
    ax.set_ylabel("Largura do Painel")
    ax.set_title("Largura do Painel vs Número de Painéis")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, max(larguras) + 0.1)

    # Build table data with cell colours. Now, first column is Número de Painéis and second is Largura.
    table_data = []
    cellColours = []
    for idx, (n, h) in enumerate(solucoes_filtradas):
        total_height = h + altura_base_pod + espessura_chao_pod
        floors = math.floor(altura_input_caminhao / total_height)
        row = [f"{n}", f"{h:.6f}m", f"{floors}"]
        table_data.append(row)
        if idx == optimal_index:
            cellColours.append(["palegreen"] * len(row))
        else:
            cellColours.append(["w"] * len(row))

    table = plt.table(cellText=table_data,
                      cellColours=cellColours,
                      colLabels=["Número de Painéis", "Largura do Painel", "Pisos no camião"],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0, -1.2, 1.5, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.subplots_adjust(left=0.2, bottom=0.3)
    plt.show()
    return solucoes_filtradas


def encontrar_solucoes_small_panels(comprimento_pod, n_min=1, n_max=10):
    """
    Calcula soluções candidatas para painéis pequenos (h = comprimento_pod / n).
    """
    solucoes = []
    for n in range(n_min, n_max+1):
        altura = round(comprimento_pod / n, 6)
        solucoes.append((n, altura))
    return solucoes

def plot_solucoes_small_panels(comprimento_pod):
    """
    Plota as soluções candidatas para os painéis pequenos e destaca a opção ótima.
    """
    solucoes = encontrar_solucoes_small_panels(comprimento_pod)
    numbers = [n for n, h in solucoes]
    alturas = [h for n, h in solucoes]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(numbers, alturas, color='purple', label="Soluções para Painéis Pequenos")
    ax.plot(numbers, alturas, color='purple', linestyle='--', alpha=0.6)
    ax.set_xlabel("Número de Painéis")
    ax.set_ylabel("Largura do Painel")
    ax.set_title("Largura do Painel vs Número de Painéis (Painéis Pequenos)")
    ax.grid(True)
    ax.legend()

    # Build table data and determine optimal candidate.
    table_data = []
    cellColours = []
    optimal = otimizar_solucao_small_panels()
    optimal_index = None
    for idx, (n, h) in enumerate(solucoes):
        total_small_height = h + altura_base_pod + espessura_chao_pod
        floors = math.floor(altura_input_caminhao / total_small_height)
        # Change the order of the data: first number of panels, then panel width, then floors.
        row = [f"{n}", f"{h:.6f}m", f"{floors}"]
        table_data.append(row)
        if optimal is not None and n == optimal[0] and abs(h - optimal[1]) < tol:
            optimal_index = idx
        cellColours.append(["w"] * len(row))
    if optimal_index is not None:
        cellColours[optimal_index] = ["palegreen"] * len(table_data[optimal_index])

    table = plt.table(cellText=table_data,
                      cellColours=cellColours,
                      colLabels=["Número de Painéis", "Largura do Painel", "Pisos no Camião"],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0, -1.2, 1.5, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.subplots_adjust(left=0.2, bottom=0.3)
    plt.show()

    return solucoes


# -------------------------------------------------------------------------------------
# ENHANCED WIDGET INTERFACE WITH TABS AND STYLING
# -------------------------------------------------------------------------------------
# Configure logging to write errors to a file and avoid console duplication.
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def input_pod_and_panel_parameters():
    # Custom CSS for widget styling
    display(HTML("""
    <style>
        .widget-label { font-weight: bold; }
        .custom-output { border: 2px solid #4CAF50; padding: 10px; background-color: #f9f9f9; }
    </style>
    """))

    style = {'description_width': '200px'}
    box_layout = widgets.Layout(width='350px', margin='10px')

    # --- POD Parameters ---
    comprimento_w = widgets.FloatText(value=2.6, description='Comprimento (m):', style=style, layout=box_layout)
    largura_w = widgets.FloatText(value=1.2, description='Largura (m):', style=style, layout=box_layout)
    altura_base_w = widgets.FloatText(value=0.10, description='Altura pé do Pod (m):', style=style, layout=box_layout)
    espessura_chao_w = widgets.FloatText(value=0.05, description='Espessura Chão (m):', style=style, layout=box_layout)
    espessura_parede_w = widgets.FloatText(value=0.06, description='Espessura da Borda (m):', style=style, layout=box_layout)
    pod_params = widgets.VBox([comprimento_w, largura_w, altura_base_w, espessura_chao_w, espessura_parede_w])

    # --- Panel Parameters ---
    altura_painel_w = widgets.FloatText(value=2.2, description='Altura Painel (m):', style=style, layout=box_layout)
    espacamento_paineis_w = widgets.FloatText(value=0.02, description='Espaçamento Painéis (m):', style=style, layout=box_layout)
    espessura_painel_w = widgets.FloatText(value=0.06, description='Espessura Painel (m):', style=style, layout=box_layout)
    espessura_painel_pequeno_w = widgets.FloatText(value=0.06, description='Espessura Painel Teto (m):', style=style, layout=box_layout)
    panel_params = widgets.VBox([altura_painel_w, espacamento_paineis_w, espessura_painel_w, espessura_painel_pequeno_w])

    # --- Sink and Toilet Parameters ---
    altura_sanita_w = widgets.FloatText(value=0.7, description='Altura Sanita (m):', style=style, layout=box_layout)
    largura_sanita_w = widgets.FloatText(value=0.4, description='Largura Sanita (m):', style=style, layout=box_layout)
    comprimento_sanita_w = widgets.FloatText(value=0.6, description='Comprimento Sanita (m):', style=style, layout=box_layout)
    altura_sink_w = widgets.FloatText(value=0.7, description='Altura Lavatório (m):', style=style, layout=box_layout)
    largura_sink_w = widgets.FloatText(value=0.4, description='Largura Lavatório (m):', style=style, layout=box_layout)
    comprimento_sink_w = widgets.FloatText(value=0.6, description='Comprimento Lavatório (m):', style=style, layout=box_layout)
    sink_toilet_params = widgets.VBox([altura_sanita_w, largura_sanita_w, comprimento_sanita_w,
                                        altura_sink_w, largura_sink_w, comprimento_sink_w])

    tab = widgets.Tab(children=[pod_params, panel_params, sink_toilet_params])
    tab.set_title(0, 'Pod')
    tab.set_title(1, 'Painéis')
    tab.set_title(2, 'Sink/Toilet')

    button = widgets.Button(description="Enviar Parâmetros", layout=widgets.Layout(width='200px', margin='20px'))
    output = widgets.Output(layout=widgets.Layout(margin='10px', border='solid 1px #ccc', padding='10px'))

    def on_button_clicked(b):
        with output:
            output.clear_output()
            try:
                # Read and validate POD parameters
                global comprimento_input_pod, largura_input_pod, altura_base_pod, espessura_chao_pod, altura_input_total_pod
                comprimento_input_pod = float(comprimento_w.value)
                largura_input_pod = float(largura_w.value)
                altura_base_pod = float(altura_base_w.value)
                espessura_chao_pod = float(espessura_chao_w.value)
                if not (0.5 <= comprimento_input_pod <= 10):
                    raise ValueError("Comprimento do pod fora do intervalo esperado (0.5 - 10 m).")
                if not (0.5 <= largura_input_pod <= 10):
                    raise ValueError("Largura do pod fora do intervalo esperado (0.5 - 10 m).")

                global espessura_parede_pod, espessura_borda_pod
                espessura_parede_pod = float(espessura_parede_w.value)

                # Read and validate Panel parameters
                global altura_painel_principal, espacamento_entre_paineis_principais, espessura_painel_principal
                global largura_input_painel_pequeno, comprimento_input_painel_pequeno
                altura_painel_principal = float(altura_painel_w.value)
                espacamento_entre_paineis_principais = float(espacamento_paineis_w.value)
                espessura_painel_principal = float(espessura_painel_w.value)
                largura_input_painel_pequeno = float(espessura_painel_pequeno_w.value)
                comprimento_input_painel_pequeno = largura_input_pod
                if altura_painel_principal <= 0 or espessura_painel_principal <= 0:
                    raise ValueError("Altura e espessura do painel devem ser maiores que zero.")

                # Set total pod height
                altura_input_total_pod = altura_base_pod + espessura_chao_w.value

                global espessura_borda_pod
                espessura_borda_pod = espessura_parede_pod

                # Read and validate Sink/Toilet parameters
                global altura_input_toilet, largura_input_toilet, comprimento_input_toilet
                global altura_input_sink, largura_input_sink, comprimento_input_sink, available_length
                altura_input_toilet = float(altura_sanita_w.value)
                largura_input_toilet = float(largura_sanita_w.value)
                comprimento_input_toilet = float(comprimento_sanita_w.value)
                altura_input_sink = float(altura_sink_w.value)
                largura_input_sink = float(largura_sink_w.value)
                comprimento_input_sink = float(comprimento_sink_w.value)
                available_length = comprimento_input_pod - altura_painel_principal - (espessura_borda_pod * 2)
                # Ensure all parameters are > 0
                params = {
                    "Comprimento Pod": comprimento_input_pod,
                    "Largura Pod": largura_input_pod,
                    "Altura pé do Pod": altura_base_pod,
                    "Espessura Chão": espessura_chao_w.value,
                    "Espessura Parede": espessura_parede_pod,
                    "Altura Painel": altura_painel_principal,
                    "Espaçamento entre Painéis": espacamento_entre_paineis_principais,
                    "Espessura Painel": espessura_painel_principal,
                    "Espessura Painel Pequeno": largura_input_painel_pequeno,
                    "Altura Sanita": altura_input_toilet,
                    "Largura Sanita": largura_input_toilet,
                    "Comprimento Sanita": comprimento_input_toilet,
                    "Altura Lavatório": altura_input_sink,
                    "Largura Lavatório": largura_input_sink,
                    "Comprimento Lavatório": comprimento_input_sink,
                }
                for nome, valor in params.items():
                    if float(valor) <= 0:
                        raise ValueError(f"{nome} deve ser maior que zero. Valor informado: {valor}")

                # Additional validations
                if comprimento_input_pod < largura_input_pod:
                    raise ValueError("O comprimento do pod deve ser maior que a largura. Trocar os valores.")
                if (comprimento_input_sink + comprimento_input_toilet) >= comprimento_input_pod:
                    raise ValueError("A soma do comprimento do sink e da sanita deve ser menor que o comprimento do pod.")
                if (comprimento_input_pod - ((2 * espessura_borda_pod) + comprimento_input_sink + comprimento_input_toilet)) <= largura_input_pod:
                    raise ValueError("As paredes do teto não cabem no pod. Irão colidir com a sanita e o lavatório (Reduzir a largura & comprimento do pod ou da sanita e lavatorio).")
                if largura_input_pod > largura_input_caminhao:
                    raise ValueError("A largura do pod é maior que a largura do caminhão. (2.4 m)")
                if comprimento_input_pod > comprimento_input_caminhao:
                    raise ValueError("O comprimento do pod é maior que o comprimento do caminhão. (13.0 m)")
                if altura_painel_principal > (comprimento_input_pod - (espessura_borda_pod * 2)):
                    raise ValueError(f"A altura dos painéis é maior que o espaço existente dentro do pod. ({comprimento_input_pod - (espessura_borda_pod * 2)} m).")

                print("Parâmetros do Pod atualizados:")
                print(f"  Comprimento: {comprimento_input_pod} m")
                print(f"  Largura: {largura_input_pod} m")
                print(f"  Altura do pé do Pod: {altura_base_pod} m")
                print(f"  Espessura do Chão: {espessura_chao_w.value} m")
                print(f"  Espessura da Parede: {espessura_parede_pod} m")
                print(f"  Altura Total do Pod: {altura_input_total_pod:.2f} m")
                print(f"  Espessura da Borda: {espessura_borda_pod} m\n")

                print("Parâmetros dos Painéis atualizados:")
                print(f"  Altura Painel: {altura_painel_principal} m")
                print(f"  Espaçamento entre Painéis: {espacamento_entre_paineis_principais} m")
                print(f"  Espessura Painel: {espessura_painel_principal} m")
                print(f"  Espessura Painel Pequeno: {largura_input_painel_pequeno} m")
                print(f"  Comprimento Painel Pequeno: {comprimento_input_painel_pequeno} m\n")

                print("Parâmetros Sink e Toilet atualizados:")
                print(f"  Altura Sanita: {altura_input_toilet} m")
                print(f"  Largura Sanita: {largura_input_toilet} m")
                print(f"  Comprimento Sanita: {comprimento_input_toilet} m")
                print(f"  Altura Lavatório: {altura_input_sink} m")
                print(f"  Largura Lavatório: {largura_input_sink} m")
                print(f"  Comprimento Lavatório: {comprimento_input_sink} m\n")

                # Continue with further processing (e.g., calling selection functions)
                selecionar_solucao_small_panels()

            except ValueError as ve:
                logging.error(ve)
                print("Erro de validação:", ve)
            except Exception as e:
                logging.error(e)
                print("Erro inesperado:", e)

    button.on_click(on_button_clicked)
    display(widgets.VBox([tab, button, output]))


# -------------------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------------------------------------------------------------

def main():
    input_pod_and_panel_parameters()

if __name__ == "__main__":
    main()



# In[ ]:




