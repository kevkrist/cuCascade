// PROTOTYPE diag: where does multi-thread per-block cost go? alloc vs free vs sync; reuse policies.
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e){std::printf("%s -> %s\n",#x,cudaGetErrorString(e));std::exit(1);}}while(0)
using clk=std::chrono::steady_clock;
static cudaMemPool_t mk(cudaMemLocationType lt, bool policies){
  cudaMemPoolProps p{}; p.allocType=cudaMemAllocationTypePinned; p.location.type=lt; p.location.id=0;
  if (lt==cudaMemLocationTypeHostNuma) p.maxSize=size_t{64}<<30;
  cudaMemPool_t pool; CK(cudaMemPoolCreate(&pool,&p)); uint64_t t=UINT64_MAX;
  CK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&t));
  if(!policies){int z=0; for(auto a:{cudaMemPoolReuseFollowEventDependencies,cudaMemPoolReuseAllowOpportunistic,cudaMemPoolReuseAllowInternalDependencies}) CK(cudaMemPoolSetAttribute(pool,a,&z));}
  if (lt==cudaMemLocationTypeHostNuma){cudaMemAccessDesc d{{cudaMemLocationTypeDevice,0},cudaMemAccessFlagsProtReadWrite}; CK(cudaMemPoolSetAccess(pool,&d,1));}
  return pool;}
// mode 0: alloc k, sync, free k (no sync). mode 1: alloc k, free k, no sync at all. mode 2: alloc k, sync, free k, sync
static void run(const char* tag, cudaMemPool_t pool,int T,int mode,bool touch){
  const size_t k=16, bs=1<<20; std::atomic<bool> stop{false}; std::atomic<long> n{0};
  std::atomic<long> ta{0},tf{0},tsy{0};
  std::vector<std::thread> w; std::vector<cudaStream_t> ss(T); for(auto&s:ss) CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  auto body=[&](int t){ CK(cudaSetDevice(0)); cudaStream_t s=ss[t]; std::vector<void*> v(k); long c=0,a=0,f=0,y=0;
    for(int it=0;!stop;++it){ auto t0=clk::now(); for(auto&p:v) CK(cudaMallocFromPoolAsync(&p,bs,pool,s)); auto t1=clk::now();
      if(mode!=1) CK(cudaStreamSynchronize(s)); auto t2=clk::now();
      if(touch&&mode!=1) for(auto p:v) *(volatile char*)p=1;
      auto t3=clk::now(); for(auto p:v) CK(cudaFreeAsync(p,s)); if(mode==2) CK(cudaStreamSynchronize(s)); auto t4=clk::now();
      if(it>=5){c+=k; a+=(t1-t0).count(); y+=(t2-t1).count(); f+=(t4-t3).count();}}
    n+=c; ta+=a; tf+=f; tsy+=y; };
  for(int t=0;t<T;++t) w.emplace_back(body,t);
  std::this_thread::sleep_for(std::chrono::milliseconds(700)); stop=true; for(auto&x:w)x.join(); CK(cudaDeviceSynchronize());
  double N=n; std::printf("  %-40s T=%2d mode=%d: alloc %7.0f  sync %7.0f  free %7.0f ns/block/thread\n",tag,T,mode,ta/N,tsy/N,tf/N);
  for(auto&s:ss) CK(cudaStreamDestroy(s));}
int main(){ CK(cudaSetDevice(0)); CK(cudaFree(0));
  for(int T:{1,8}) for(int mode:{0,1,2}){
    {auto p=mk(cudaMemLocationTypeHostNuma,true); run("hostnuma, default reuse policies",p,T,mode,true); CK(cudaMemPoolDestroy(p));}
    {auto p=mk(cudaMemLocationTypeHostNuma,false); run("hostnuma, reuse policies OFF",p,T,mode,true); CK(cudaMemPoolDestroy(p));}
    {auto p=mk(cudaMemLocationTypeDevice,true); run("DEVICE pool (control)",p,T,mode,false); CK(cudaMemPoolDestroy(p));}
  }}
