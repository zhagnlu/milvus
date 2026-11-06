from pathlib import Path
exec(Path('/tmp/delete_opt_kernel_check.py').read_text().split("Path('/tmp/delete_opt_kernel_check.cpp')")[0])
bench=r'''
void bench(size_t n,size_t m,size_t chunks,bool dense){
 std::vector<int64_t> rows(n);for(size_t i=0;i<n;++i)rows[i]=i*2;
 std::mt19937_64 rng(53901);
 std::vector<PkType> pks(m);for(size_t i=0;i<m;++i)pks[i]=int64_t(dense ? n+2*i : rng()%(2*n));
 std::sort(pks.begin(),pks.end());
 uint64_t total=0;
 auto get_timestamp=[](size_t){return Timestamp(1);};
 auto read_ts=[](int64_t){return Timestamp(0);};
 auto timestamp_hit=[](Timestamp a,Timestamp b){return a<b;};
 auto callback=[&](SegOffset o,Timestamp t){total+=o.n+t;};
 const size_t repeat=std::max(size_t(2),std::min(size_t(300),size_t(10000)/m));
 auto run=[&](bool merge){
  auto begin=std::chrono::steady_clock::now();
  for(size_t r=0;r<repeat;++r){
   // KERNEL
   for(size_t c=0;c<chunks;++c){
    size_t first=c*n/chunks,last=(c+1)*n/chunks;
    if(merge){
     search_sorted_chunk(last-first,first,[&](size_t i){return rows[first+i];},[&](size_t i){return std::get<int64_t>(pks[i]);});
    }else{
     for(auto& pk:pks){
      auto k=std::get<int64_t>(pk);
      auto it=std::lower_bound(rows.begin()+first,rows.begin()+last,k);
      for(;it!=rows.begin()+last && *it==k;++it)callback(SegOffset(it-rows.begin()),1);
     }
    }
   }
  }
  return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/repeat;
 };
 double old=0,now=0;
 for(int r=0;r<4;++r){if(r%2){now+=run(true);old+=run(false);}else{old+=run(false);now+=run(true);}}
 printf("rows=%zu batch=%zu chunks=%zu %s binary_us=%.2f merge_us=%.2f speedup=%.2f checksum=%llu\n",n,m,chunks,dense?"dense":"random",old/4,now/4,old/now,(unsigned long long)total);
}
int main(){for(size_t n:{2000000,44000000})for(size_t chunks:{1,8})for(bool dense:{false,true})for(size_t m:{1,8,128,4096,100000})bench(n,m,chunks,dense);}
'''
Path('/tmp/delete_opt_lookup_bench.cpp').write_text(preamble+bench.replace('   // KERNEL',kernel))
