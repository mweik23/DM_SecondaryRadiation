      SUBROUTINE ML5_0_MP_BORN_AMPS_AND_WFS(P)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     Computes all the AMP and WFS in quadruple precision for the 
C     phase space point P(0:3,NEXTERNAL)
C     
C     Process: g g > t t~ QCD<=2 QED=0 [ virt = QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=3)
      INTEGER    NLOOPAMPS, NCTAMPS
      PARAMETER (NLOOPAMPS=129, NCTAMPS=85)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10)
      INTEGER    NCOMB
      PARAMETER (NCOMB=16)
      REAL*16     ZERO
      PARAMETER (ZERO=0E0_16)
      COMPLEX*32 IMAG1
      PARAMETER (IMAG1=(0E0_16,1E0_16))

C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J,H
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC/NEXTERNAL*1/
C     
C     FUNCTIONS
C     
      LOGICAL ML5_0_IS_HEL_SELECTED
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl_same_name.inc'

      INTEGER NTRY
      LOGICAL CHECKPHASE,HELDOUBLECHECKED
      REAL*8 REF
      COMMON/ML5_0_INIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

      LOGICAL GOODHEL(NCOMB)
      LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER HELPICKED
      COMMON/ML5_0_HELCHOICE/HELPICKED

      COMPLEX*32 AMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_MP_AMPS/AMP
      COMPLEX*16 DPAMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_AMPS/DPAMP
      COMPLEX*32 W(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_MP_WFS/W

      COMPLEX*32 AMPL(3,NCTAMPS)
      COMMON/ML5_0_MP_AMPL/AMPL

      COMPLEX*16 DPW(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_WFCTS/DPW

      COMPLEX*16 DPAMPL(3,NLOOPAMPS)
      LOGICAL S(NLOOPAMPS)
      COMMON/ML5_0_AMPL/DPAMPL,S

      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/ML5_0_HELCONFIGS/HELC

      LOGICAL MP_DONE_ONCE
      COMMON/ML5_0_MP_DONE_ONCE/MP_DONE_ONCE

C     This array specify potential special requirements on the
C      helicities to
C     consider. POLARIZATIONS(0,0) is -1 if there is not such
C      requirement.
      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/ML5_0_BEAM_POL/POLARIZATIONS

C     ----------
C     BEGIN CODE
C     ---------

      MP_DONE_ONCE=.TRUE.

C     To be on the safe side, we always update the MP params here.
C     It can be redundant as this routine can be called a couple of
C      times for the same PS point during the stability checks.
C     But it is really not time consuming and I would rather be safe.
      CALL MP_UPDATE_AS_PARAM()

      DO H=1,NCOMB
        IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1)
     $   .AND.((CHECKPHASE.OR..NOT.HELDOUBLECHECKED).OR.GOODHEL(H))))
     $    THEN
C         Handle the possible requirement of specific polarizations
          IF ((.NOT.CHECKPHASE)
     $     .AND.HELDOUBLECHECKED.AND.POLARIZATIONS(0,0)
     $     .EQ.0.AND.(.NOT.ML5_0_IS_HEL_SELECTED(H))) THEN
            CYCLE
          ENDIF
          DO I=1,NEXTERNAL
            NHEL(I)=HELC(I,H)
          ENDDO
          CALL MP_VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1,H))
          CALL MP_VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2,H))
          CALL MP_OXXXXX(P(0,3),MDL_MT,NHEL(3),+1*IC(3),W(1,3,H))
          CALL MP_IXXXXX(P(0,4),MDL_MT,NHEL(4),-1*IC(4),W(1,4,H))
          CALL MP_VVV1P0_1(W(1,1,H),W(1,2,H),GC_4,ZERO,ZERO,W(1,5,H))
C         Amplitude(s) for born diagram with ID 1
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),GC_5,AMP(1,H))
          CALL MP_FFV1_1(W(1,3,H),W(1,1,H),GC_5,MDL_MT,MDL_WT,W(1,6,H))
C         Amplitude(s) for born diagram with ID 2
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),GC_5,AMP(2,H))
          CALL MP_FFV1_2(W(1,4,H),W(1,1,H),GC_5,MDL_MT,MDL_WT,W(1,7,H))
C         Amplitude(s) for born diagram with ID 3
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),GC_5,AMP(3,H))
          CALL MP_FFV1P0_3(W(1,4,H),W(1,3,H),GC_5,ZERO,ZERO,W(1,8,H))
C         Counter-term amplitude(s) for loop diagram number 4
          CALL MP_R2_GG_1_0(W(1,5,H),W(1,8,H),R2_GGQ,AMPL(1,1))
          CALL MP_R2_GG_1_0(W(1,5,H),W(1,8,H),R2_GGQ,AMPL(1,2))
          CALL MP_R2_GG_1_0(W(1,5,H),W(1,8,H),R2_GGQ,AMPL(1,3))
          CALL MP_R2_GG_1_0(W(1,5,H),W(1,8,H),R2_GGQ,AMPL(1,4))
C         Counter-term amplitude(s) for loop diagram number 5
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,5))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,6))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,7))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,8))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB,AMPL(1,9))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,10))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GT,AMPL(1,11))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GB_1EPS,AMPL(2
     $     ,12))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),UV_3GG_1EPS,AMPL(2
     $     ,13))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,14))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,15))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,16))
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,17))
C         Counter-term amplitude(s) for loop diagram number 7
          CALL MP_R2_GG_1_R2_GG_3_0(W(1,5,H),W(1,8,H),R2_GGQ,R2_GGB
     $     ,AMPL(1,18))
C         Counter-term amplitude(s) for loop diagram number 8
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,19))
C         Counter-term amplitude(s) for loop diagram number 10
          CALL MP_R2_GG_1_R2_GG_3_0(W(1,5,H),W(1,8,H),R2_GGQ,R2_GGT
     $     ,AMPL(1,20))
C         Counter-term amplitude(s) for loop diagram number 11
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,21))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,22))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,23))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,24))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB,AMPL(1,25))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,26))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQT,AMPL(1,27))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQB_1EPS
     $     ,AMPL(2,28))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),UV_GQQG_1EPS
     $     ,AMPL(2,29))
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),R2_GQQ,AMPL(1,30))
          CALL MP_FFV1_2(W(1,4,H),W(1,2,H),GC_5,MDL_MT,MDL_WT,W(1,9,H))
C         Counter-term amplitude(s) for loop diagram number 13
          CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,9,H),W(1,6,H),R2_QQQ,R2_QQT
     $     ,AMPL(1,31))
          CALL MP_R2_QQ_2_0(W(1,9,H),W(1,6,H),UV_TMASS,AMPL(1,32))
          CALL MP_R2_QQ_2_0(W(1,9,H),W(1,6,H),UV_TMASS_1EPS,AMPL(2,33))
C         Counter-term amplitude(s) for loop diagram number 14
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,34))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,35))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,36))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,37))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB,AMPL(1,38))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,39))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQT,AMPL(1,40))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,41))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),UV_GQQG_1EPS
     $     ,AMPL(2,42))
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),R2_GQQ,AMPL(1,43))
          CALL MP_FFV1_1(W(1,3,H),W(1,2,H),GC_5,MDL_MT,MDL_WT,W(1,10,H)
     $     )
C         Counter-term amplitude(s) for loop diagram number 16
          CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,7,H),W(1,10,H),R2_QQQ,R2_QQT
     $     ,AMPL(1,44))
          CALL MP_R2_QQ_2_0(W(1,7,H),W(1,10,H),UV_TMASS,AMPL(1,45))
          CALL MP_R2_QQ_2_0(W(1,7,H),W(1,10,H),UV_TMASS_1EPS,AMPL(2,46)
     $     )
C         Counter-term amplitude(s) for loop diagram number 17
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,47))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,48))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,49))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,50))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB,AMPL(1,51))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,52))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQT,AMPL(1,53))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQB_1EPS
     $     ,AMPL(2,54))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),UV_GQQG_1EPS
     $     ,AMPL(2,55))
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),R2_GQQ,AMPL(1,56))
C         Counter-term amplitude(s) for loop diagram number 19
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,57))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,58))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,59))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,60))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB,AMPL(1,61)
     $     )
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,62))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQT,AMPL(1,63)
     $     )
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,64))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),UV_GQQG_1EPS
     $     ,AMPL(2,65))
          CALL MP_FFV1_0(W(1,4,H),W(1,10,H),W(1,1,H),R2_GQQ,AMPL(1,66))
C         Counter-term amplitude(s) for loop diagram number 20
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,67))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,68))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,69))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,70))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB,AMPL(1,71))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,72))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQT,AMPL(1,73))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQB_1EPS
     $     ,AMPL(2,74))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),UV_GQQG_1EPS
     $     ,AMPL(2,75))
          CALL MP_FFV1_0(W(1,9,H),W(1,3,H),W(1,1,H),R2_GQQ,AMPL(1,76))
C         Counter-term amplitude(s) for loop diagram number 22
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GQ,AMPL(1,77))
C         Counter-term amplitude(s) for loop diagram number 32
          CALL MP_R2_GG_1_R2_GG_2_0(W(1,5,H),W(1,8,H),R2_GGG_1
     $     ,R2_GGG_2,AMPL(1,78))
C         Counter-term amplitude(s) for loop diagram number 33
          CALL MP_VVV1_0(W(1,1,H),W(1,2,H),W(1,8,H),R2_3GG,AMPL(1,79))
C         Amplitude(s) for UVCT diagram with ID 40
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),GC_5,AMPL(2,80))
          AMPL(2,80)=AMPL(2,80)*(4.0D0*UVWFCT_G_1_1EPS+2.0D0
     $     *UVWFCT_B_0_1EPS)
C         Amplitude(s) for UVCT diagram with ID 41
          CALL MP_FFV1_0(W(1,4,H),W(1,3,H),W(1,5,H),GC_5,AMPL(1,81))
          AMPL(1,81)=AMPL(1,81)*(2.0D0*UVWFCT_T_0+2.0D0*UVWFCT_G_1
     $     +2.0D0*UVWFCT_G_2)
C         Amplitude(s) for UVCT diagram with ID 42
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),GC_5,AMPL(2,82))
          AMPL(2,82)=AMPL(2,82)*(4.0D0*UVWFCT_G_1_1EPS+2.0D0
     $     *UVWFCT_B_0_1EPS)
C         Amplitude(s) for UVCT diagram with ID 43
          CALL MP_FFV1_0(W(1,4,H),W(1,6,H),W(1,2,H),GC_5,AMPL(1,83))
          AMPL(1,83)=AMPL(1,83)*(2.0D0*UVWFCT_T_0+2.0D0*UVWFCT_G_1
     $     +2.0D0*UVWFCT_G_2)
C         Amplitude(s) for UVCT diagram with ID 44
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),GC_5,AMPL(2,84))
          AMPL(2,84)=AMPL(2,84)*(4.0D0*UVWFCT_G_1_1EPS+2.0D0
     $     *UVWFCT_B_0_1EPS)
C         Amplitude(s) for UVCT diagram with ID 45
          CALL MP_FFV1_0(W(1,7,H),W(1,3,H),W(1,2,H),GC_5,AMPL(1,85))
          AMPL(1,85)=AMPL(1,85)*(2.0D0*UVWFCT_T_0+2.0D0*UVWFCT_G_1
     $     +2.0D0*UVWFCT_G_2)
C         Copy the qp wfs to the dp ones as they are used to setup the
C          CT calls.
          DO I=1,NWAVEFUNCS
            DO J=1,20
              DPW(J,I,H)=W(J,I,H)
            ENDDO
          ENDDO
C         Same for the counterterms amplitudes
          DO I=1,NCTAMPS
            DO J=1,3
              DPAMPL(J,I)=AMPL(J,I)
              S(I)=.TRUE.
            ENDDO
          ENDDO
          DO I=1,NBORNAMPS
            DPAMP(I,H)=AMP(I,H)
          ENDDO
        ENDIF
      ENDDO

      END

