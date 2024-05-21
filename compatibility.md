## 互換性

この章では、2つの重要な「互換性」の概念について説明します。

- cuDNN APIの互換性: 他のバージョンのcuDNNを使用して構築されたアプリケーションとの前方互換性と後方互換性に関するもの
- cuDNNハードウェアの前方互換性: 特定のcuDNNバージョンが将来のハードウェアと互換性があるかどうかに関するもの

この章の残りの部分では、これらの概念について詳しく説明します。

### cuDNN APIの互換性

cuDNN 7以降、パッチおよびマイナーリリースのバイナリ互換性は次のように維持されます：

- 任意のパッチリリース x.y.z は、他のcuDNNパッチリリース x.y.w で構築されたアプリケーションと前方および後方互換性があります（つまり、同じメジャーおよびマイナーバージョン番号を持つが、w!=zである場合）。
- cuDNNのマイナーリリースは、同じまたは以前のマイナーリリースで構築されたアプリケーションとバイナリ後方互換性があります（つまり、cuDNN x.y は、cuDNN x.z で構築されたアプリとバイナリ互換性があります。ここで z<=y）。
- cuDNNバージョンx.zでコンパイルされたアプリケーションは、z>yの場合、x.yリリースと互換性があることは保証されません。

### cuDNNハードウェアの前方互換性

cuDNNのビルドが将来のハードウェアで機能的に動作する場合、それはハードウェアの前方互換性があります。これにより、cuDNNにリンクされたアプリケーションは、cuDNNの新しいリリースに更新することなく、将来のハードウェアで動作し続けることができます。ただし、いくつかの注意点があります：

- この機能はcuDNN 9.0.0で追加され、CUDAツールキット12以上を使用するビルドでのみ適用されます。それ以前のバージョンのcuDNNはハードウェアの前方互換性がありません。
- cuDNNには、ハードウェアの前方互換性にいくつかの制限があります。これらの制限は、このセクションの後半で説明されています。

このセクションでは、特に明記されていない限り、「前方互換性」や「互換性」という用語の使用は、ハードウェアの前方互換性を意味します。

各リリースで、cuDNNライブラリはネイティブにサポートするSMバージョンのリストを持っています。これらはcuDNNサポートマトリックスに記載されています。これらのSMバージョンのいずれかで実行される場合、ライブラリはネイティブモードで動作します。ネイティブサポートとは、特定のSMアーキテクチャ用にコンパイルされた明示的なCUDAカーネルを含むライブラリを意味します。

一方、ネイティブにサポートされているものより新しいデバイスで実行される場合、ライブラリは前方互換性モードで動作し、PTX JITコンパイルを使用してライブラリのワークロードをサポートします。

**注意**

各バージョンのcuDNNライブラリは、ネイティブにサポートするGPUの最高のSM番号を保存します。これは通常、cuDNNバージョンがリリースされた時点で生産中の最新のNVIDIA GPUのSM番号です。この値は、関数`cudnnGetMaxDeviceVersion()`を呼び出して照会できます。この番号よりも高いSM番号を持つGPUについては、そのSMは前方互換性モードでサポートされます。

#### 前方互換性モードの設定

デフォルトでは、ライブラリがネイティブにサポートしていない新しいGPUで実行されていることを検出した場合、自動的に前方互換性モードが有効になります。それが望ましくない場合、環境変数`CUDNN_FORWARD_COMPAT_DISABLE=1`をエクスポートして前方互換性モードを無効にします。

その場合、ライブラリはネイティブにサポートしていない将来のデバイスで失敗する可能性があります。

将来のハードウェアデバイスのサポートが必要な場合は、次のいずれかを推奨します：

- 新しいハードウェアでネイティブにサポートされるバージョンのcuDNNライブラリにアップグレードします（これにより、新しいハードウェアでの最良のサポートが提供されます）。
- 前方互換性サポートをデフォルトで有効にしておきます（つまり、`CUDNN_FORWARD_COMPAT_DISABLE=0`またはエクスポートしない）。これにより、ライブラリはデフォルトでNVIDIA GPUの将来のアーキテクチャを前方互換性モードでサポートしようとします。

### 前方互換性とグラフAPI

一般的な原則として、現在のハードウェアで機能するcuDNNグラフAPIの使用は、将来のハードウェアでも機能するはずですが、すべてのグラフパターンが現在サポートされているわけではありません（サポートされているグラフパターンのセクションを参照）。

つまり、エンジンIDやエンジン設定ノブの直接指定のような高度な使用であっても、前方互換性モードの動作を妨げることはありません。必要に応じて、ライブラリは非前方互換性のエンジン設定を前方互換性のあるエンジン設定に置き換えます（詳細は後述します）。

ただし、可能な限り最高のパフォーマンスを得るために、ヒューリスティックスワークフローを推奨します。これは前方互換性モードでなくても推奨されるフローであり、ほとんどのユースケースで問題になることはないと予想されます。

次のサブセクションでは、さまざまなワークフローの詳細を説明します。

#### ヒューリスティックスベースのワークフロー

このユースケースは次の条件が真である場合に適用されます：

- 前方互換性モードで操作グラフの最速のエンジン設定を望む。
- 新しいデバイスで初めて実行する際にcuDNNヒューリスティックスをクエリすることをいとわない。
- ネイティブモードで以前に実行していたエンジン以外のエンジンを実行しても問題ない。

この場合、ユーザーは常にcuDNNヒューリスティックス（特に`CUDNN_HEUR_MODE_A`、現在他のモードは前方互換性向けに調整されていないため）を呼び出して、操作グラフに推奨されるエンジン設定のリストをクエリし、その中から選択することを推奨します。これは、ユーザーが以前に自動調整されたエンジン設定やその他のエンジン設定を新しいデバイスに初めて切り替える際に使用しないことを意味します。

前方互換性モードでは、`CUDNN_HEUR_MODE_A`には次の重要な特性があります：

- 前方互換性モードでヒューリスティックスが呼び出された場合、前方互換性のないエンジン設定を返しません。
- 前方互換性モード向けに特別に調整されており、よりパフォーマンスの高い設定を提供できます。

前方互換性モードを無視しても、現在のcuDNNのヒューリスティックスは、返されるエンジン設定のリストが与えられた入力問題に対して確定することを保証しません。最悪の場合、ヒューリスティックスクエリから返されたエンジン設定のいずれもが与えられた入力問題に対して確定しない可能性があります。ユーザーはそのような場合をキャッチし、`CUDNN_HEUR_MODE_FALLBACK`モードのヒューリスティックスからフォールバック設定を使用して適切に処理する必要があります。前方互換性モードでは、このヒューリスティックスの動作は一貫しており、ユーザーはヒューリスティックスクエリから返されたエンジン設定のいずれもが確定できない場合を処理するようにアプリケーションを構築する必要があります。

ヒューリスティックスが前方互換性モードでクエリされた場合とネイティブモードでクエリされた場合で、推奨されるエンジン設定が異なる可能性があるため、推奨されるエンジン設定の数値的特性やパフォーマンスを2つのモード間で比較することはできません。ユーザーは、推奨されるエンジンの動作ノートをクエリし、ネイティブモードと同様に、前方互換性モードで望まない数値的特性を持つエンジンをフィルタリングすることを期待されます。

#### ヒューリスティックスベースのワークフローに従わない場合の特記事項

前述のように、ヒューリスティックスワークフローは推奨されますが、前方互換性サポートを得るために必須ではありません。現在ヒューリスティックスワークフローに従っていない場合、コード変更を望まない場合は、コードやcuDNNへの呼び出し方法を変更する必要はなく、以前に動作していたグラフAPI呼び出しがシームレスに動作し続けるはずです。

前方互換性サポートが有効になると、cuDNNライブラリはネイティブモードでサポートされているより新しいSMで実行されていることを自動的に検出し、前方互換性モードを有効にします。ライブラリは、ネイティブにサポートされているGPUで実行される場合には存在しないエラーをすべてキャッチしようとします。例えば、ユーザーが古いアーキテクチャに特化したエンジン設定を実行しようとし、それが新しいアーキテクチャでサポートされていない場合などです。このようなエラーが発生した場合、ライブラリはエンジン設定を別のエンジン設定に置き換え、操作が成功するようにします。

エンジン設定が前方互換性エラーに遭遇した場合、ライブラリがエンジン設定を置き換えるため、以前のエンジン設定に関連していたノブは無視されます。基本的に、ライブラリはユーザーのエンジン設定の選択を最大限尊重しますが、前方互換性の問題でその設定が失敗した場合、ライブラリにはエンジン設定を前方互換性のある設定に置き換えるためのエラーハンドリングメカニズムが組み込まれており、以前の操作グラフが常に成功し続けることを保証します。これにより、エラーがユーザーアプリケーションに伝播しないようにします。

現在、置き換えられたエンジン設定はネイティブにサポートされているエンジン設定と同等のパフォーマンスを提供することはできませんが、機能は損なわれません。

前方互換性モードはPTX JIT（Just In Time）コンパイルに依存するため、コンパイルオーバーヘッドが発生する可能性があります。CUDAアプリケーションと同様に、これらのオーバーヘッドは遅延読み込み（例えば、CUDA_MODULE_LOADING環境変数を参照）とJITコンパイル済みカーネルのキャッシュ（JITキャッシュに関する詳細はこのCUDAセクションを参照）で管理できます。

### サポートされているグラフパターン

cuDNNで正常に確定できるすべての操作グラフを前方互換性にすることが目標ですが、現時点ではそうではありません。以下は、このリリース時点で前方互換性のあるグラフ操作のセットです：

- 事前コンパイルされた単一操作エンジン
  - これには、ConvolutionFwd、ConvolutionBwdFilter、およびConvolutionBwdData、またはConvolutionBwBias、NormalizationForward、およびNormalizationBackwardが含まれます。
- 汎用ランタイム融合エンジン
- 特殊な事前コンパイルエンジン
  - ConvBiasAct
  - ConvScaleBiasAct

上記に記載されていない操作グラフはまだ前方互換性がありません。ただし、後続のリリースで前方互換性が追加される予定です。現在サポートされていないグラフパターンの注目すべき例としては次のものがあります：

- 融合された注意および融合されたフラッシュ注意（fpropおよびbprop）のグラフパターンの特殊なランタイム融合エンジン
- BnAddRelu、DReluForkDBnのグラフパターンの特殊な事前コンパイルエンジン
- FP8融合フラッシュ注意

汎用ランタイム融合エンジン内で次のケースに対する前方互換性モードに既知の問題があります：

- FP8を使用する任意の汎用ランタイム融合エンジン。
- 次のパターンを含むいくつかの畳み込み融合：
  - 汎用ランタイム融合エンジン内でのCUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTORを含むConvolution Forward操作の後に続く操作のg 2は、完全なテンソルへのブロードキャストのみをサポートし、ベクトルまたはスカラーのテンソル間のブロードキャストはサポートしません。後者を使用すると、結果が正しくない可能性があります。
  - グループ化された畳み込み（G>1の畳み込み）を含む融合パターンはサポートされていません。
  - 汎用ランタイム融合エンジン内でのCUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTORを含むConvolution Forward操作の後に続く操作のg 2は、次の削減パターンをサポートしません：
    - [N, K, P, Q] -> [N, 1, 1, 1]
    - [N, K, P, Q] -> [N, K, 1, 1]
  - CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTORを組み合わせたg 1内の融合パターンは、CUDNN_POINTWISE_MUL（ポイントワイズ：スケール）、CUDNN_POINTWISE_ADD（ポイントワイズ：バイアス）、CUDNN_POINTWISE_RELU_FWD（ポイントワイズ：Relu）モードを含むもので、畳み込みForward / ConvolutionBackwardFilterについてはfloat32のデータタイプのみをサポートし、ConvolutionBackwardDataは任意のデータタイプではサポートされません。
  - 混合精度の入力テンソルを持つMatMul融合はサポートされていません。

### 前方互換性とレガシーAPI

現在、レガシーAPIのほとんどは前方互換性があります。例えば、次の一般的に使用されるレガシーAPIルーチンは前方互換性があります：

- `cudnnConvolutionForward()`
- `cudnnConvolutionBackwardData()`
- `cudnnConvolutionBackwardFilter()`

次に、まだ前方互換性がないレガシーAPIルーチンの包括的なリストを示します。その他のレガシーAPIの使用はすべて前方互換性があります。

#### RNN（リカレントニューラルネットワーク）

- `cudnnRNNForward()`
- `cudnnRNNBackwardData_v8()`
- `cudnnRNNBackwardWeights_v8()`

#### バッチ正規化

- `cudnnBatchNormalizationBackward()`
- `cudnnBatchNormalizationBackwardEx()`
- `cudnnBatchNormalizationForwardInference()`
- `cudnnBatchNormalizationForwardTraining()`
- `cudnnBatchNormalizationForwardTrainingEx()`

#### 融合

- `cudnnFusedOpsExecute()`と次の値を持つ`cudnnFusedOps_t` opsパラメータ：
  - `CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS`
  - `CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING`
  - `CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE`
  - `CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK`
  - `CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM`

レガシーAPIの一部が非推奨になっていることも注目に値しますが、これは前方互換性には影響しません。上記のリストは、レガシーAPI内で前方互換性のないものの真実のソースです。