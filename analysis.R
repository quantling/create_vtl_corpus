

# zipf-plots

wav_orig <- dir('vtl_corpus1.0/wav_original/')
tokens_orig <- tools::file_path_sans_ext(wav_orig)  # remove .wav
tokens_orig <- substring(tokens_orig, 8)  # remove numbers in the beginning

types_orig <- table(tokens_orig)
types_orig <- sort(types_orig, decreasing=T)

wav_synth <- dir('vtl_corpus1.0/wav_synthesized/')
tokens_synth <- tools::file_path_sans_ext(wav_synth)  # remove .wav
tokens_synth <- substring(tokens_synth, 8)  # remove numbers in the beginning

types_synth <- table(tokens_synth)
types_synth <- sort(types_synth, decreasing=T)

pdf('figs/zipf.pdf', height=4, width=8)
par(mfcol=c(1,2))

plot(log10(types_orig), log='x', axes=F, xlab="words", ylab="occurrences", main='Recorded')
axis(1, at=1:length(types_orig), labels=names(types_orig))
yticks <- c(1:9, 1:9*10, 1:9*100, 1:2*1000) 
axis(2, at=log10(yticks), labels=yticks)

plot(log10(types_synth), log='x', axes=F, xlab="words", ylab="occurrences", main='Synthesized')
axis(1, at=1:length(types_synth), labels=names(types_synth))
yticks <- c(1:9, 1:9*10, 1:9*100, 1:2*1000) 
axis(2, at=log10(yticks), labels=yticks)
dev.off()


# durations

dur_orig <- rep(NA, length(wav_orig))

for (ii in 1:length(wav_orig)) {
	name <- wav_orig[ii]
	wav <- readWave(paste0('vtl_corpus1.0/wav_original/', name))
	dur_orig[ii] <- length(wav@left) / wav@samp.rate
}

dur_synth <- rep(NA, length(wav_synth))

for (ii in 1:length(wav_synth)) {
	name <- wav_synth[ii]
	wav <- readWave(paste0('vtl_corpus1.0/wav_synthesized/', name))
	dur_synth[ii] <- length(wav@left) / wav@samp.rate
}


