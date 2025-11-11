import mujoco
import os

# Coloque o nome do seu arquivo XML aqui
xml_file_name = "inverted_pendulum.xml"  # Mude para o path/nome correto

# Encontra o caminho absoluto do arquivo
script_dir = os.path.dirname(__file__)
xml_path = os.path.join(script_dir, xml_file_name)

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Erro ao carregar o XML: {e}")
    print(f"Verifique se o arquivo '{xml_file_name}' está na mesma pasta que este script.")
    exit()

print("--- Constantes Físicas do Modelo ---")

# 1. Pega a massa do carrinho (M)
try:
    cart_body_id = model.body("cart").id
    M = model.body_mass[cart_body_id]
    print(f"Massa do Carrinho (M): {M:.4f} kg")
except KeyError:
    print("Erro: Não foi possível encontrar o body 'cart' no XML.")

# 2. Pega a massa do pêndulo (mp)
try:
    pole_body_id = model.body("pole").id
    mp = model.body_mass[pole_body_id]
    print(f"Massa do Pêndulo (mp): {mp:.4f} kg")
except KeyError:
    print("Erro: Não foi possível encontrar o body 'pole' no XML.")

print("\n--- Constantes Adicionais ---")
print(f"Gravidade (g): {model.opt.gravity[2] * -1}") # Pega o -9.81 e inverte
print("Comprimento do Pêndulo (L): 0.3 m (calculado do 'fromto' 0->0.6)")