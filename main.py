# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    if not isinstance(n,int) or n<=0:
        return None
    return np.cos(np.arange(0,n)*np.pi/(n-1))



def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    if not isinstance(n,int) or n<=0:
        return None
    w=np.ones(n)
    w[0]=0.5
    w[-1]=(-1)**(n-1)*0.5
    w[1:-1:2]=-1
    return w


def barycentric_inte(
    xi: np.ndarray,
    yi: np.ndarray,
    wi: np.ndarray,
    x: np.ndarray
) -> np.ndarray | None:
    

    
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray):
        return None
    if not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None

    if xi.ndim != 1 or yi.ndim != 1 or wi.ndim != 1 or x.ndim != 1:
        return None

    if len(xi) != len(yi) or len(xi) != len(wi):
        return None

    m = xi.size
    result = np.zeros_like(x, dtype=float)

    for idx, xv in enumerate(x):
        dx = xv - xi

        
        hit = np.nonzero(dx == 0)[0]
        if hit.size:
            result[idx] = yi[hit[0]]
            continue

        temp = wi / dx
        result[idx] = np.dot(temp, yi) / np.sum(temp)

    return result




def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
        return abs(xr - x)
    
    return np.max(np.abs(np.array(xr) - np.array(x)))

