import os
import sys
import subprocess
import time

def clear_screen():
    # Clears the terminal screen for a clean look
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("="*60)
    print("EMSI - PROJET DE FIN DE MODULE : DEEP LEARNING")
    print("Réalisé par : BOUDANE Taha")
    print("Année Universitaire : 2025-2026")
    print("="*60)
    print("\nCe tableau de bord permet d'exécuter et d'évaluer les trois")
    print("architectures développées (MLP, CNN, RNN/Seq2Seq).\n")

def run_script(script_name, title):
    print_header()
    print(f"Lancement de la {title}...\n")
    print(f"Fichier cible : {script_name}")
    print("-" * 60 + "\n")
    
    if not os.path.exists(script_name):
        print(f"Erreur : Le fichier '{script_name}' est introuvable.")
        print("Vérifiez qu'il se trouve dans le même dossier que ce script.")
    else:
        try:
            # Runs the script using the current Python executable (your dl environment)
            subprocess.run([sys.executable, script_name], check=True)
        except subprocess.CalledProcessError:
            print(f"\nUne erreur s'est produite lors de l'exécution de {script_name}.")
        except KeyboardInterrupt:
            print(f"\nExécution interrompue par l'utilisateur.")
            
    print("\n" + "-" * 60)
    input("Appuyez sur [ENTRÉE] pour retourner au menu principal...")

def main_menu():
    while True:
        print_header()
        print("Veuillez sélectionner le module à présenter :\n")
        print("  [1] Partie I   : Classification Tabulaire avec MLP (Wine Quality)")
        print("  [2] Partie II  : Vision par Ordinateur avec CNN (Fashion-MNIST)")
        print("  [3] Partie III : Traduction Seq2Seq avec GRU (Fra-Eng)")
        print("  [4] Quitter")
        print("\n" + "="*60)
        
        choice = input("Votre choix (1-4) : ").strip()
        
        if choice == '1':
            run_script('part1_mlp_tabular.py', "Partie I (MLP)")
        elif choice == '2':
            run_script('part2_cnn_vision.py', "Partie II (CNN)")
        elif choice == '3':
            run_script('part3_rnn_nlp.py', "Partie III (RNN/Seq2Seq)")
        elif choice == '4':
            clear_screen()
            print("Fermeture du tableau de bord. Excellente présentation ! \n")
            break
        else:
            print("\nChoix invalide. Veuillez entrer 1, 2, 3 ou 4.")
            time.sleep(1.5)

if __name__ == "__main__":
    main_menu()