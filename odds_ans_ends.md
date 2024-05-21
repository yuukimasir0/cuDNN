
## 雑多な事項

このセクションには、さまざまなトピックや概念が含まれています。

### スレッドセーフ

cuDNNライブラリはスレッドセーフです。その関数は、同じcuDNNハンドルを同時に共有しない限り、複数のホストスレッドから呼び出すことができます。

スレッドごとにcuDNNハンドルを作成する場合、各スレッドが自分のハンドルを非同期に作成する前に、最初に`cudnnCreate()`の単一の同期呼び出しを行うことを推奨します。

複数のスレッドから同じデバイスを使用するマルチスレッドアプリケーションの場合、推奨されるプログラミングモデルは、スレッドごとに1つ（または必要に応じて複数）のcuDNNハンドルを作成し、そのcuDNNハンドルをスレッドの全期間にわたって使用することです。

### 再現性（決定論）

設計上、特定のバージョンのほとんどのcuDNNルーチンは、同じアーキテクチャのGPU上で実行された場合、ランごとに同じビット単位の結果を生成します。いくつかの例外があります。たとえば、次のルーチンは、同じアーキテクチャ上でも実行ごとに再現性を保証しません。これは、真にランダムな浮動小数点の丸め誤差を引き起こす方法でアトミック操作を使用するためです：

- `CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0`または`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3`が使用される場合の`cudnnConvolutionBackwardFilter`
- `CUDNN_CONVOLUTION_BWD_DATA_ALGO_0`が使用される場合の`cudnnConvolutionBackwardData`
- `CUDNN_POOLING_MAX`が使用される場合の`cudnnPoolingBackward`
- `cudnnSpatialTfSamplerBackward`
- `CUDNN_CTC_LOSS_ALGO_NON_DETERMINSTIC`が使用される場合の`cudnnCTCLoss`および`cudnnCTCLoss_v8`

異なるアーキテクチャ間では、cuDNNルーチンはビット単位の再現性を保証しません。たとえば、NVIDIA VoltaとNVIDIA Turingアーキテクチャで同じルーチンを比較する場合、ビット単位の再現性は保証されません。

### スケーリングパラメータ

`cudnnConvolutionForward()`などの多くのcuDNNルーチンは、ホストメモリ内のスケーリング係数`alpha`および`beta`へのポインタを受け入れます。これらのスケーリング係数は、計算された値を目的のテンソル内の以前の値とブレンドするために使用されます：

```cpp
dstValue = alpha * computedValue + beta * priorDstValue
```

dstValueは読み取り後に書き込まれます。

畳み込みのスケーリングパラメータ
`beta`がゼロの場合、出力は読み取られず、未初期化のデータ（NaNを含む）が含まれている可能性があります。

これらのパラメータは、ホストメモリポインタを使用して渡されます。alphaおよびbetaのストレージデータ型は次のとおりです：

`HALF`および`FLOAT`テンソルの場合は`float`
`DOUBLE`テンソルの場合は`double`
パフォーマンスを向上させるには、`beta = 0.0`を使用します。出力テンソルの現在の値と以前の値をブレンドする必要がある場合にのみ、betaにゼロ以外の値を使用します。

型変換
データ入力`x`、フィルタ入力`w`、および出力`y`がすべてINT8データ型の場合、関数`cudnnConvolutionBiasActivationForward()`は型変換を行います。アキュムレータはオーバーフロー時にラップする32ビット整数です。

cpp
コードをコピーする
INT8のcudnnConvolutionBiasActivationForward
非推奨ポリシー
cuDNNは、すべてのAPIおよびenumの変更に対して、迅速なイノベーションを可能にするために、簡略化された2段階の非推奨ポリシーを使用します：

ステップ1：非推奨のラベル付け
現在のメジャーバージョンでは、動作を変更せずにAPI関数またはenumを非推奨としてマークします。

非推奨のenum値は、CUDNN_DEPRECATED_ENUMマクロでマークされます。
単に名前が変更された場合、古い名前は新しい名前にマッピングされ、古い名前はCUDNN_DEPRECATED_ENUMマクロでマークされます。
非推奨のAPI関数は、CUDNN_DEPRECATEDマクロでマークされます。
ステップ2：削除
次のメジャーバージョンでは、非推奨のAPI関数またはenum値が削除され、その名前は再利用されません。

この非推奨スキームにより、非推奨のAPIを1つのメジャーリリースで廃止できます。現在のメジャーリリースで非推奨となった機能は、変更なしでコンパイルできます。次のメジャーcuDNNリリースが導入されると、後方互換性は終了します。

非推奨の関数のプロトタイプは、CUDNN_DEPRECATEDマクロを使用してcuDNNのヘッダーに追加されます。-DCUDNN_WARN_DEPRECATEDスイッチをコンパイラに渡すと、コード内の非推奨関数呼び出しがコンパイラ警告を発します。たとえば：

scss
コードをコピーする
警告： 'cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t, cudnnRNNClipMode_t, ...)' は非推奨です [-Wdeprecated-declarations]
または

arduino
コードをコピーする
警告C4996： 'cudnnRNNSetClip_v8'： 非推奨として宣言されました
上記の警告は、コンパイラ警告がエラーとして扱われるソフトウェアセットアップでのビルドの中断を避けるためにデフォルトで無効にされています。

同様に、非推奨のenum値についても、非推奨値を使用しようとするとコンパイラが警告を発します：

arduino
コードをコピーする
警告： 'EXAMPLE_VAL' は非推奨です： 値が許可されていません [-Wdeprecated-declarations]
または

arduino
コードをコピーする
警告C4996： 'EXAMPLE_VAL'： 非推奨として宣言されました
特別なケース：APIの動作変更
開発者への驚きを避けるために、特定のAPI関数の2つのメジャーバージョン間の動作変更は、関数に_vタグを付け、現在のメジャーcuDNNバージョンに続く形式で対応します。次のメジャーリリースでは、非推奨の関数が削除され、その名前は再利用されません（新しいAPIはまず_vタグなしで導入されます）。

このように関数の動作を更新することで、API呼び出しが変更されたcuDNNバージョンをAPIの名前に埋め込むことができます。その結果、APIの変更は追跡および文書化が容易になります。

cuDNNの2つの連続したメジャーリリース、バージョン8および9を使用した例を使用して、このプロセスを説明します。この例では、API関数foo()の動作がcuDNN v7からcuDNN v8に変更されます。

メジャーリリース8
更新されたAPIはfoo_v8()として導入されます。後方互換性を維持するために、非推奨のAPI foo()は次のメジャーリリースまで変更されません。

メジャーリリース9
非推奨のAPI foo()は永久に削除され、その名前は再利用されません。foo_v8()関数は、廃止されたfoo()呼び出しに取って代わります。

GPUとドライバの要件
最新の互換性ソフトウェアバージョン、OS、CUDA、CUDAドライバ、およびNVIDIAハードウェアについては、cuDNNサポートマトリックスを参照してください。

畳み込みのための規約と機能
畳み込み関数は次のとおりです：

cudnnConvolutionBackwardData()
cudnnConvolutionBiasActivationForward()
cudnnConvolutionForward()
cudnnConvolutionBackwardBias()
cudnnConvolutionBackwardFilter()
畳み込みの公式
このセクションでは、cudnnConvolutionForward()パスのcuDNN畳み込み関数で実装されているさまざまな畳み込み公式について説明します。

次の表に示す畳み込み用語は、後に続くすべての畳み込み公式に適用されます。

畳み込み用語
用語	説明
x	入力（画像）テンソル
w	ウェイトテンソル
y	出力テンソル
n	現在のバッチサイズ
c	現在の入力チャネル
C	総入力チャネル
H	入力画像の高さ
W	入力画像の幅
k	現在の出力チャネル
K	総出力チャネル
p	現在の出力の高さ位置
q	現在の出力の幅位置
G	グループ数
pad	パディング値
u	垂直方向のサブサンプルストライド（高さ方向）
v	水平方向のサブサンプルストライド（幅方向）
dil h	垂直方向のダイレーション（高さ方向）
dil w	水平方向のダイレーション（幅方向）
r	現在のフィルタの高さ
R	総フィルタの高さ
s	現在のフィルタの幅
S	総フィルタの幅
CuDNN_CROSS_CORRELATIONモードに設定された畳み込み
パディング付き畳み込み
サブサンプルストライド付き畳み込み
ダイレーション付き畳み込み
CuDNN_CONVOLUTIONモードに設定された畳み込み
グループ化畳み込みを使用した畳み込み
グループ化畳み込み
cuDNNは、畳み込みディスクリプタconvDescに対してcudnnSetConvolutionGroupCount()を使用してgroupCount > 1を設定することで、グループ化畳み込みをサポートします。デフォルトでは、畳み込みディスクリプタconvDescはgroupCount 1に設定されています。

基本的な考え方
概念的には、グループ化畳み込みでは、入力チャネルとフィルタチャネルが独立したグループのgroupCount数に分割され、各グループにはチャネル数が減少します。その後、これらの入力グループとフィルタグループに対して別々に畳み込み操作が実行されます。例えば、次のように考えてください：入力チャネルの数が4で、フィルタチャネルの数が12の場合。通常の非グループ化畳み込みでは、実行される計算操作の数は12*4です。

groupCountが2に設定されている場合、2つの入力チャネルグループ（各2つの入力チャネルを含む）と2つのフィルタチャネルグループ（各6つのフィルタチャネルを含む）が存在します。その結果、各グループ化された畳み込みは2*6の計算操作を実行し、2つのグループ化された畳み込みが実行されます。したがって、計算の節約は2倍：（12*4）/（2*（2*6））。

cuDNNグループ化畳み込み
グループ化畳み込みにgroupCountを使用する場合でも、テンソルディスクリプタをすべて定義して、グループごとのサイズではなく、畳み込み全体のサイズを説明する必要があります。

グループ化畳み込みは、cudnnConvolutionForward()、cudnnConvolutionBackwardData()、およびcudnnConvolutionBackwardFilter()の関数で現在サポートされているすべての形式に対してサポートされています。

groupCount 1に設定されたテンソルのストライドも、任意のグループカウントに対して有効です。

デフォルトでは、畳み込みディスクリプタconvDescはgroupCount 1に設定されています。cuDNNのグループ化畳み込みの数学については、畳み込みの公式を参照してください。

例
以下に、2D畳み込みの場合のNCHW形式のグループ化畳み込みの次元とストライドを示します。記号*および/は乗算および除算を示すために使用されます。

xDescまたはdxDesc
次元：[batch_size、input_channel、x_height、x_width]
ストライド：[input_channels * x_height * x_width、x_height * x_width、x_width、1]
wDescまたはdwDesc
次元：[output_channels、input_channels / groupCount、w_height、w_width]
形式：NCHW
convDesc
グループ数：groupCount
yDescまたはdyDesc
次元：[batch_size、output_channels、y_height、y_width]
ストライド：[output_channels * y_height * y_width、y_height * y_width、y_width、1]
3D畳み込みのベストプラクティス
注意: これらのガイドラインは、NVIDIA cuDNN v7.6.3以降の3D畳み込みおよび逆畳み込み関数に適用されます。

以下のガイドラインは、3D畳み込みのパフォーマンスを向上させるためにcuDNNライブラリのパラメータを設定するためのものです。具体的には、フィルタサイズ、パディング、およびダイレーション設定などの設定に焦点を当てています。さらに、医療画像処理というアプリケーション固有のユースケースが提示され、これらの推奨設定を使用した3D畳み込みのパフォーマンス向上が示されています。

具体的には、以下の関数およびそれらに関連するデータ型に適用されます：

cudnnConvolutionForward()
cudnnConvolutionBackwardData()
cudnnConvolutionBackwardFilter()
詳細については、cuDNN APIリファレンスを参照してください。

推奨設定
次の表は、cuDNNで3D畳み込みを実行する際の推奨設定を示しています。

プラットフォーム	畳み込み（3Dまたは2D）	畳み込みまたは逆畳み込み（fprop、dgrad、またはwgrad）	グループ化畳み込みのサイズ	データレイアウト形式（NHWC / NCHW）	I / O精度（FP16、FP32、INT8、またはFP64）	アキュムレータ（計算）精度（FP16、FP32、INT32、またはFP64）	フィルタ（カーネル）サイズ	パディング	画像サイズ	Cチャネルの数	Kチャネルの数	畳み込みモード	ストライド	ダイレーション	データポインタのアライメント
NVIDIA Hopperアーキテクチャ	3Dおよび2D	fprop	C_per_group == K_per_group == {1,4,8,16,32,64,128,256}	NDHWC	FP16	FP32	制限なし	制限なし	テンソルの2GB制限	0 mod 8	0 mod 8	相関および畳み込み	制限なし	制限なし	すべてのデータポインタは16バイトにアライメントされています。
NVIDIA Ampereアーキテクチャ	3Dおよび2D	dgrad	C_per_group == K_per_group == {1,4,8,16,32,64,128,256}	NDHWC	FP32 - NVIDIA AmpereアーキテクチャのデフォルトのTF32数学。CUDNN_TENSOROP_MATH_ALLOW_CONVERSIONプレアンペア。	INT32	制限なし	制限なし	テンソルの2GB制限	0 mod 8	0 mod 8	相関および畳み込み	制限なし	制限なし	すべてのデータポインタは16バイトにアライメントされています。
NVIDIA Turingアーキテクチャ	3Dおよび2D	wgrad	C_per_group == K_per_group == {1,4,8,16,32,64,128,256}	NDHWC	INT8 - INT8はdgradおよびwgradをサポートしていません。INT8 3D畳み込みはバックエンドAPIでのみサポートされています。	INT32	制限なし	制限なし	テンソルの2GB制限	0 mod 8	0 mod 8	相関および畳み込み	制限なし	制限なし	すべてのデータポインタは16バイトにアライメントされています。
NVIDIA Voltaアーキテクチャ	3Dおよび2D	fprop, dgrad, wgrad	C_per_group == K_per_group == {1,4,8,16,32,64,128,256}	NDHWC	INT8 - INT8はdgradおよびwgradをサポートしていません。INT8 3D畳み込みはバックエンドAPIでのみサポートされています。	INT32	制限なし	制限なし	テンソルの2GB制限	0 mod 8	0 mod 8	相関および畳み込み	制限なし	制限なし	すべてのデータポインタは16バイトにアライメントされています。
制限事項
モデルにチャネル数が32未満の場合、パフォーマンスが低下する可能性があります（低くなるほど悪化します）。ネットワークに上記が含まれている場合、cuDNNFind*を使用して最適なオプションを取得してください。

環境変数
cuDNNの動作は、一連の環境変数を通じて影響を受ける可能性があります。次の環境変数はcuDNNによって公式にサポートされています：

NVIDIA_TF32_OVERRIDE
CUDNN_LOGDEST_DBG
CUDNN_LOGLEVEL_DBG
CUDNN_LOGINFO_DBG（非推奨）
CUDNN_LOGWARN_DBG（非推奨）
CUDNN_LOGERR_DBG（非推奨）
CUDNN_FORWARD_COMPAT_DISABLE
これらの変数の詳細については、cuDNN APIリファレンスを参照してください。

注意

上記の環境変数を除いて、CUDNN_で始まる他の環境変数の使用に関しては、サポートや保証は提供されません。

SMカーブアウト
cuDNN 8.9.5以降、NVIDIA Hopper GPUでSMカーブアウトがサポートされており、エキスパートユーザーは別のCUDAストリームで同時実行するためにSMを予約することができます。ユーザーは、cuDNNヒューリスティックスに目標SM数を設定し、その数のSMを使用して実行するエンジン設定のリストを取得できます。cuDNNヒューリスティックスを使用せずに高度なユースケースでは、SMカーブアウトを設定してエンジン設定を最初から作成することもできます（この機能をサポートするエンジンは以下の表に記載されています）。

以下のコードスニペットは、ヒューリスティックスのユースケースのサンプルです。

cpp
コードをコピーする
// ヒューリスティックスディスクリプタを作成
cudnnBackendDescriptor_t engHeur;
cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &engHeur);
cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph);
cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heurMode);
// SMカーブアウト
int32_t targetSMCount = 66;
cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET, CUDNN_TYPE_INT32, 1, &targetSMCount);
cudnnBackendFinalize(engHeur);
// エンジン設定ディスクリプタを作成
cudnnBackendDescriptor_t engConfig;
cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engConfig);
// ヒューリスティックスから最適なエンジン設定を取得
cudnnBackendGetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returnedCount, engConfig);
// "engConfig" は目標SM数66で準備ができました
この機能は、通常の畳み込み（Fprop、Dgrad、Wgrad）およびConv-Bias-Act融合でサポートされています。

cuDNNバックエンドエンジンでSMカーブアウトをサポートしているもの
畳み込みフォワード
畳み込みバックワードデータ
畳み込みバックワードフィルタ
cudnnConvolutionBiasActivationForward
バージョンチェックとCUDNN_VERSION
CUDNN_VERSIONの定義は次のとおりです：

cpp
コードをコピーする
CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL
CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL
したがって、CUDNN_VERSIONを使用するバージョンチェックは、適宜更新する必要があります。たとえば、ユーザーがcuDNNが9.0.0以上の場合にのみコードパスを実行したい場合、CUDNN_VERSION >= 90000ではなく、CUDNN_VERSION >= 9000のマクロ条件を使用する必要があります。

cuDNNシンボルサーバ
アプリケーションでデバッグまたはプロファイリングされているcuDNNライブラリの難読化されたシンボルは、Linuxのシンボルリポジトリからダウンロードできます。リポジトリには、難読化されたシンボル名を含むシンボルファイル（.sym）がホストされています（デバッグデータは配布されていません）。

cuDNN APIで問題が発生した場合、そのスタックトレースにシンボルサーバを使用することでデバッグプロセスの迅速化が可能です。

GNU Debugger（GDB）で各cuDNNライブラリの難読化されたシンボルを使用するための推奨方法は次の2つです：

ライブラリのアンストリップ
.symファイルを別のデバッグ情報ファイルとしてデプロイ
以下のコードは、x86_64 Ubuntu 22.04で難読化されたシンボルを使用する推奨方法を示しています：

sh
コードをコピーする
# ライブラリのビルドIDを確認
$ readelf -n /usr/lib/x86_64-linux-gnu/libcudnn_graph.so

# ... ビルドID: 457c8f5dea095b0f90af2abddfcb69946df61b76
# シンボルファイル名を決定するためにhttps://cudatoolkit-symbols.nvidia.com/libcudnn_graph.so/457c8f5dea095b0f90af2abddfcb69946df61b76/index.htmlにアクセス
$ wget https://cudatoolkit-symbols.nvidia.com/libcudnn_graph.so/457c8f5dea095b0f90af2abddfcb69946df61b76/libcudnn_graph.so.9.0.0.sym

# 適切な権限を持って、アンストリップするか、
$ eu-unstrip /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.0.0 libcudnn_graph.so.9.0.0.sym -o /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.0.0

# または、適切な権限を持って別のデバッグ情報ファイルとしてデプロイ
# ビルドIDを2つの部分に分割し、最初の2文字をディレクトリとして使用
# 残りの文字を.debug拡張子を持つファイル名として使用
$ cp libcudnn_graph.so.9.0.0.sym /usr/lib/debug/.build-id/45/7c8f5dea095b0f90af2abddfcb69946df61b76.debug
シンボル化の例
シンボル化の使用例を簡単に示します。test_sharedという名前のサンプルアプリケーションがcuDNN APIのcudnnDestroy()を呼び出し、セグメンテーションフォルトを引き起こす場合、デフォルトのcuDNNインストールと難読化されたシンボルがない場合のGDBの出力は次のようになります：

sh
コードをコピーする
Thread 1 "test_shared" received signal SIGSEGV, Segmentation fault.
0x00007ffff7a4ac01 in ?? () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
(gdb) bt
#0  0x00007ffff7a4ac01 in ?? () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
#1  0x00007ffff7a4c919 in cudnnDestroy () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
#2  0x00000000004007b7 in main ()
前述の方法のいずれかを使用して難読化されたシンボルを適用した後、スタックトレースは次のようになります：

sh
コードをコピーする
Thread 1 "test_shared" received signal SIGSEGV, Segmentation fault.
0x00007ffff7a4ac01 in libcudnn_graph_148ced18265f5231d89551dcbdcf5cf3fe6d77d1 () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
(gdb) bt
#0  0x00007ffff7a4ac01 in libcudnn_graph_148ced18265f5231d89551dcbdcf5cf3fe6d77d1 () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
#1  0x00007ffff7a4c919 in cudnnDestroy () from /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
#2  0x00000000004007b7 in main ()
シンボル化されたコールスタックは、NVIDIAに提供されるバグ記述の一部として文書化できます。
