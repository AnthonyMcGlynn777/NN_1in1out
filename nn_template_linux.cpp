/* 1入力1出力のニューラルネットワーク */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUT 2
#define NUM_HIDDEN 600			/* 隠れ層のユニット数 */
#define RAND (double)rand()/(double)RAND_MAX	/* 区間[0,1]の疑似一様乱数 */
#define RAND255 (int)rand()%256		/*区間[0,255]の疑似一様乱数*/
#define ETA 0.01			/* η(イータ)：学習係数(小さい正の値) */
#define TRAINING_TIMES 129400			/* 学習回数 */
#define f(z) 1.0/(1.0+exp(-z))		/* シグモイド関数 */
#define MAX_DATA 20					/* 教師データの最大数（これより多く設定したい場合変更する */

/* プロトタイプ宣言 */
double NN(double x, double v[][NUM_INPUT + 1], double w[]);

int main(void)
{
	int t;

	double x[2];				/* 入力値 */
	x[1] = 1.0;					/* 閾値ユニットへの入力は常に１ */

	double v[NUM_HIDDEN][NUM_INPUT + 1];	/* 入力から隠れ層ユニットへの結合荷重（入力と閾値の2つに対する重み）*/
	double u_h[NUM_HIDDEN];		/* 隠れ層ユニットの内部状態 */
	double z[NUM_HIDDEN + 1];	/*  隠れ層ユニットの出力 */
	z[NUM_HIDDEN] = 1.0;		/*  閾値ユニットへの入力は常に１ */

	double w[NUM_HIDDEN + 1];	/*  隠れ層から出力ユニットへの結合荷重（閾値ユニット含むため+1） */
	double u_y = 0;					/*  出力ユニットの内部状態 */
	double y;					/*  出力ユニットの出力 */
	double yd;					/*  目標出力（教師信号） */


	double delta_y;				/* 誤差逆伝搬に用いる変数 */
	double delta_h[NUM_HIDDEN];	/* 誤差逆伝搬に用いる変数 */

	double error = 0;				/* ある一つの入力に対する誤差 */
	double sum_error = 0;			/* すべてのデータに対する誤差　*/

	double data[MAX_DATA][2];	/*  入力データ（行：データ番号，列：入力xと出力yのペアつまり教師データ） */

	srand(1825050703);			/*  カッコ内の乱数シードを学籍番号にしてください） */

	double temp;
	int num_data = 0;

	FILE* f1 = fopen("training_data.txt", "r");
	FILE* f2 = fopen("Traning_Times.txt", "w");
	FILE* f3 = fopen("Error_Data.txt", "w");
	FILE* f4 = fopen("error.txt", "w");
	FILE* f5 = fopen("Test_Data.txt", "w");
	FILE* f6 = fopen("Test_Result.txt", "w");

	/*  訓練データ（training_data.txt）の読込み */
	printf("Reading training_data.txt\n");

	while (fscanf(f1, "%lf", &temp) != EOF) {
		data[num_data][0] = temp / 255.0;	/*  0～255までを0～1.0の範囲に収めるように正規化 */
		printf("%f\t", temp);				/* （ニューラルネットワークでは入力値を±1.0付近に収めることが多い */
		
		fscanf(f1, "%lf", &temp);
		data[num_data][1] = temp / 100.0;	/*  0～100を0～1.0の範囲に収めるように正規化 */
		printf("%f\n", temp);

		num_data++;							/*  訓練データ数をのカウント（training_data.txtの行数カウント） */
	}
	fclose(f1);
	printf("Press Enter to continue...");
	getchar();								/* キー入力待ち */

	/*  【学習ここから】 */

	/*  【初期化】結合重みの初期化（0±1.0の範囲でランダムに重みを初期化している） */
	for (int j = 0; j < NUM_HIDDEN; j++) {
		for (int i = 0; i < NUM_INPUT + 1; i++) {
			v[j][i] = 2.0 * RAND - 1.0;							/*  入力xに対する重み */
		}
	}
	for (int j = 0; j < NUM_HIDDEN + 1; j++) {
		w[j] = 2.0 * RAND - 1.0;								/*  中間層ユニットと閾値ユニットに対する重み */
	}

	/*  【出力計算と学習のループ（TRAINING_TIMES回繰り返す）】 */
	for (t = 0; t < TRAINING_TIMES; t++) {

		sum_error = 0.0; /* 全データに対する誤差の和計算用 */
		for (int data_count = 0; data_count < num_data; data_count++) {

			x[0] = data[data_count][0];
			u_y = 0.0;

			/* 【入力から出力を計算する】 */
			/* 入力から中間層の出力zを計算（実習）*/
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				u_h[j] = 0;
				for(int i = 0; i < NUM_INPUT+1; i++){
					u_h[j] += x[i] * v[j][i];
				}
				z[j] = f(u_h[j]);
			}

			/* 中間層から出力層の出力yを計算（実習） */
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				u_y += w[j] * z[j];
			}
			y = u_y;											/*  NNの出力（シグモイド関数を通さない） */
			yd = data[data_count][1];							/*  目標出力（教師信号） */

			/* 誤差を計算（実習） */

			error = (1.0/2.0) * pow(y-yd, 2.0);
			sum_error += error;

			/* 【誤差逆伝搬法による学習】 */
			/*  出力層のデルタdelta_yの計算（実習） */
			delta_y = (y - yd);
			
			/*  中間層から出力層の重み・閾値更新（実習） */
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				w[j] += -ETA * delta_y * z[j];
			}

			/*  中間層のデルタdelta_h[j]の計算（実習） */
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				delta_h[j] = (y-yd) * z[j] * (1-z[j]) * w[j];
			}

			/*  入力層から中間層の重み・閾値更新（実習） */
			for(int j = 0; j < NUM_HIDDEN; j++){
				for(int i = 0; i < NUM_INPUT+1; i++){
					v[j][i] += -ETA * delta_h[j] * x[i]; 
				}
			}



			/*  【誤差逆伝搬法ここまで】 */

		}
		/* 訓練回数と誤差和をprintfで表示する．同時にfprintfでファイルにも保存する（実習） */
		printf("%d %lf\n",t,sum_error);
		fprintf(f2, "%d\n", t+1);
		fprintf(f3, "%lf\n",sum_error);
		fprintf(f4, "%d		%lf\n", t+1, sum_error);

	}
	fclose(f2);
	fclose(f3);
	fclose(f4);

	/* テスト誤差の検証（実習）*/
	printf("Press Enter to start testing...");
	getchar();													/* キー入力待ち */

	for(int i=0; i<255; i++){
		x[0] = (double)(RAND255) / 255.0;
		u_y = 0;
	/* 【入力から出力を計算する】 */
		/* 入力から中間層の出力zを計算*/
		for(int j = 0; j < NUM_HIDDEN+1; j++){
			u_h[j] = 0;
			for(int i = 0; i < NUM_INPUT+1; i++){
				u_h[j] += x[i] * v[j][i];
			}
			z[j] = f(u_h[j]);
		}

		/* 中間層から出力層の出力yを計算 */
		for(int j = 0; j < NUM_HIDDEN+1; j++){
			u_y += w[j] * z[j];
		}
		y = u_y * 100;											/*  NNの出力（シグモイド関数を通さない） */
		printf("%lf\n",y);
		fprintf(f5, "%.2lf\n", x[0]*255);
		fprintf(f6, "%lf\n", y);
	}
	fclose(f5);
	fclose(f6);

	return 0;
}