# EDSM
Embedded Deformation for Shape Manipulation (Newton / LM / PCG / CG)

함수최적화 기법을 EDSM(doi.org/10.1145/1276377.1276478)을 바탕으로 matlab을 통해 구현
(두 mesh가 있을 때, 하나의 위치에서 다른 위치로 mesh를 이동)
https://darkpgmr.tistory.com/142를 통해 함수최적화 기법 더 알아볼 수 있습니다.

PCG(Preconditioned Conjugate Gradient Method)는 아래 링크 참조
https://en.wikipedia.org/wiki/Conjugate_gradient_method#Convergence_properties
Preconditioner로는 주로 'none': just vanilla Jacobian $\times$ Jacobian (let Jp) / 'eye': add eye of Jp / 'diag': add twice diagonal of Jp사용
