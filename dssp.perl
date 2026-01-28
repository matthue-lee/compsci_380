sub Prot_Uni_Calc_DSSP {
      my $dssp_folder = "$work_dir/DSSP";
      system("rm -rf $dssp_folder");
      system("mkdir $dssp_folder");
      system("mkdir $dssp_folder/SecondaryStructure");
      system("rm -rf $work_dir/Fasta_Files");
      system("mkdir $work_dir/Fasta_Files");
      my $pdb_folder = "$work_dir/PDB_Files";

      my @split_1 = ();
      my @split_2 = ();


      while(defined(my $pdb = glob("$pdb_folder/*.pdb"))){
            my @split_1 = split(/\// , $pdb);
            my @split_2 = split(/\./, $split_1[-1]);
            system("$dssp $pdb > $dssp_folder/$split_1[-1].dssp");
      }
      
      while(defined(my $in = glob("$dssp_folder/*.dssp"))) {
            my @split_1 = split(/\// , $in);
            my @split_2 = split(/\./, $split_1[-1]);


            open(IN, "< $in");
            my $trigger = 0;
            my @aa_fasta = ();
            my @dssp_fasta =();
            my $cnt = 0;
            while(defined(my $line = <IN>)){
#                 chomp($line);
                  if ($trigger == 1){
                        my @split_ln_1 = split(//, $line);
#                       print "$split_ln_1[13]\t$split_ln_1[16]\n";
                        $aa_fasta[$cnt] = $split_ln_1[13];
                        if ($split_ln_1[16] =~ /\s/){
                              $dssp_fasta[$cnt] = "-";
                        }
                        else{
                              $dssp_fasta[$cnt] = $split_ln_1[16];
                        }
                        $cnt++;
                  }
                  elsif ($line =~ /\s+\#\s+/) {$trigger = 1}
            }
            close(IN);
#           print "@aa_fasta\n";
#           print "@dssp_fasta\n";
            
            open(BIG_OUT, ">> $analysis_root/Analysis/DSSP/DSSP_outputs.txt");
            print BIG_OUT "$split_2[0]\t$cnt\t";
            open(OUT_DSSP, "> $dssp_folder/SecondaryStructure/$split_2[0].ss");
            open(OUT_AA, "> $work_dir/Fasta_Files/$split_2[0].fa");
            print OUT_DSSP "> $split_2[0]\n";
            print OUT_AA "> $split_2[0]\n";
            my $t = 0;
            
      #SS_types
            my $H = 0;
            my $B = 0;
            my $E = 0;
            my $G = 0;
            my $I = 0;
            my $T = 0;
            my $S = 0;
            
            while($t < (scalar @dssp_fasta)){
                  print OUT_AA "$aa_fasta[$t]"; 
                  print OUT_DSSP "$dssp_fasta[$t]"; 
                  if ($dssp_fasta[$t] =~ /H/){
                        $H++;
                  }
                  elsif ($dssp_fasta[$t] =~ /B/){
                        $B++
                  }
                  elsif ($dssp_fasta[$t] =~ /E/){
                        $E++
                  }
                  elsif ($dssp_fasta[$t] =~ /G/){
                        $G++
                  }           
                  elsif ($dssp_fasta[$t] =~ /I/){
                        $I++
                  }           
                  elsif ($dssp_fasta[$t] =~ /T/){
                        $T++
                  }           
                  elsif ($dssp_fasta[$t] =~ /S/){
                        $S++
                  }                 
                  $t++;
            }
            print OUT_DSSP "\n";
            print OUT_AA "\n";
            close(OUT_DSSP);
            close(OUT_AA);
            my $total_helix_percent = ($H+$G+$I) / $cnt;
            my $total_beta_percent = ($B+$E) / $cnt;
            my $total_big_secondary_percent = ($H+$G+$I+$B+$E) / $cnt;
            print BIG_OUT "$H\t$B\t$E\t$G\t$I\t$T\t$S\t$total_helix_percent\t$total_beta_percent\t$total_big_secondary_percent\n";
            close (BIG_OUT);